"""
固定 CBR JPEG+LDPC 评估脚本
在与 DeepJSCC ld=512 相同的 CBR=0.167 下运行 JPEG+LDPC，提供最严格的公平对比。

CBR=0.167 → channel_uses=512 → 信息比特=512 (LDPC k=512, n=1024 中只传1个码字)
                                → JPEG 压缩预算 = 512 bits = 64 bytes
用法：
    python eval_jpeg_fixed_cbr.py
"""

import os, json, io, math
import numpy as np
import tensorflow as tf

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')

import sionna
from sionna.channel import AWGN
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import ebnodb2no, BinarySource
from PIL import Image

SNR_POINTS = [-5, 0, 5, 10, 15, 20, 25]
NUM_IMAGES  = 500
PATCH_SIZE  = 32

# 固定 CBR=0.167：与 DeepJSCC ld=512 相同
# channel_uses = 512 (一个 QAM-16 码字, 256 符号对应 1024 bits, 其中 512 信息 bits)
K_BITS   = 512   # LDPC 信息比特
N_BITS   = 1024  # LDPC 编码比特
CODERATE = K_BITS / N_BITS          # 0.5
NUM_BITS_PER_SYMBOL = 4             # QAM-16
CHANNEL_USES = N_BITS // NUM_BITS_PER_SYMBOL  # 256
SOURCE_SAMPLES = PATCH_SIZE * PATCH_SIZE * 3  # 3072
CBR = CHANNEL_USES / SOURCE_SAMPLES           # 0.0833... wait let me recalc

# CBR = channel_uses / source_samples = 256 / 3072 = 0.0833
# DeepJSCC ld=512 CBR = 512 / 3072 = 0.1667
# For CBR=0.1667 exactly: channel_uses = 512 → N_BITS = 2048, K_BITS = 1024
# Let's use K=1024, N=2048, rate=0.5 → channel_uses = 2048/4 = 512 → CBR=512/3072=0.1667
K_BITS   = 1024
N_BITS   = 2048
CODERATE = K_BITS / N_BITS
CHANNEL_USES = N_BITS // NUM_BITS_PER_SYMBOL   # 512
CBR = CHANNEL_USES / SOURCE_SAMPLES            # 0.1667
JPEG_BUDGET_BYTES = K_BITS // 8               # 128 bytes

print(f"Fixed-CBR JPEG+LDPC config:")
print(f"  LDPC: k={K_BITS}, n={N_BITS}, rate={CODERATE:.1f}")
print(f"  Modulation: QAM-{2**NUM_BITS_PER_SYMBOL}")
print(f"  Channel uses per image: {CHANNEL_USES}")
print(f"  CBR: {CBR:.4f} (matches DeepJSCC ld=512: {512/3072:.4f})")
print(f"  JPEG budget: {JPEG_BUDGET_BYTES} bytes = {K_BITS} bits\n")


def jpeg_encode_fixed(img_uint8, max_bytes):
    """Binary search for highest JPEG quality within byte budget."""
    lo, hi = 1, 95
    best_buf = None
    for _ in range(10):
        q = (lo + hi) // 2
        buf = io.BytesIO()
        Image.fromarray(img_uint8).save(buf, format='JPEG', quality=q)
        if buf.tell() <= max_bytes:
            best_buf = buf
            lo = q + 1
        else:
            hi = q - 1
    if best_buf is None:
        buf = io.BytesIO()
        Image.fromarray(img_uint8).save(buf, format='JPEG', quality=1)
        best_buf = buf
    best_buf.seek(0)
    return best_buf.read()


def bytes_to_bits(data, n_bits):
    """Pad/truncate bytes to exactly n_bits."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)[:n_bits]
    if len(bits) < n_bits:
        bits = np.pad(bits, (0, n_bits - len(bits)))
    return bits.astype(np.float32)


def bits_to_bytes(bits, n_bytes):
    bits_int = bits[:n_bytes * 8].astype(np.uint8)
    if len(bits_int) < n_bytes * 8:
        bits_int = np.pad(bits_int, (0, n_bytes * 8 - len(bits_int)))
    return np.packbits(bits_int).tobytes()


def simulate_jpeg_awgn_fixed(img_uint8, ebno_db):
    # 1. JPEG compress within budget
    jpeg_data = jpeg_encode_fixed(img_uint8, JPEG_BUDGET_BYTES)

    # 2. Pad/truncate to K_BITS
    bits = bytes_to_bits(jpeg_data, K_BITS)
    bits_tf = tf.constant(bits[np.newaxis, :], dtype=tf.float32)

    # 3. LDPC encode
    encoder = LDPC5GEncoder(k=K_BITS, n=N_BITS)
    codeword = encoder(bits_tf)

    # 4. QAM-16 modulate
    constellation = Constellation('qam', NUM_BITS_PER_SYMBOL)
    mapper = Mapper(constellation=constellation)
    symbols = mapper(codeword)  # shape (1, 512)

    # 5. AWGN channel
    no = ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE)
    channel = AWGN()
    rx = channel([symbols, no])

    # 6. Demodulate
    demapper = Demapper('app', constellation=constellation)
    llrs = demapper([rx, no])

    # 7. LDPC decode
    decoder = LDPC5GDecoder(encoder, hard_out=True)
    bits_hat = decoder(llrs)  # (1, K_BITS)

    # 8. Reconstruct JPEG
    bits_np = bits_hat.numpy()[0].astype(np.uint8)
    jpeg_out = bits_to_bytes(bits_np, JPEG_BUDGET_BYTES)
    try:
        recon = np.array(Image.open(io.BytesIO(jpeg_out)).convert('RGB').resize(
            (PATCH_SIZE, PATCH_SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
        orig  = img_uint8.astype(np.float32) / 255.0
        mse   = float(np.mean((orig - recon) ** 2))
        psnr  = float(10 * math.log10(1.0 / mse)) if mse > 0 else 40.0
        ssim  = float(tf.reduce_mean(tf.image.ssim(
            tf.constant(orig[np.newaxis]), tf.constant(recon[np.newaxis]), max_val=1.0)))
        return psnr, ssim, True
    except Exception:
        return 0.0, 0.0, False


def main():
    (_, _), (x_test_raw, _) = tf.keras.datasets.cifar10.load_data()
    imgs = x_test_raw[:NUM_IMAGES]

    results = []
    for snr in SNR_POINTS:
        psnrs, ssims, successes = [], [], []
        for img in imgs:
            psnr, ssim, ok = simulate_jpeg_awgn_fixed(img, snr)
            psnrs.append(psnr)
            ssims.append(ssim)
            successes.append(ok)

        sr = float(np.mean(successes))
        avg_psnr = float(np.mean(psnrs)) if sr > 0.5 else 0.0
        avg_ssim = float(np.mean(ssims)) if sr > 0.5 else 0.0
        entry = {'ebno_db': snr, 'psnr': avg_psnr, 'ssim': avg_ssim,
                 'success_rate': sr, 'cbr': CBR}
        results.append(entry)
        print(f"SNR={snr:+3d}  PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}  SR={sr:.2f}")

    out = 'checkpoints/image-jscc/eval/jpeg_fixed_cbr.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}  (CBR={CBR:.4f}, JPEG budget={JPEG_BUDGET_BYTES} bytes)')


if __name__ == '__main__':
    main()

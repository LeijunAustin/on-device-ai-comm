"""
Traditional JPEG + AWGN channel baseline
Fair comparison: same SNR conditions, same test set
"""
import os, json
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from sionna.channel import AWGN
from sionna.utils import ebnodb2no, BinarySource
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
import io

def jpeg_encode(img_uint8, quality=75):
    """Encode image to JPEG bytes."""
    img_pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()

def jpeg_decode(buf):
    """Decode JPEG bytes to image."""
    img_pil = Image.open(io.BytesIO(buf))
    return np.array(img_pil)

def bytes_to_bits(data):
    """Convert bytes to bit array."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.float32)

def bits_to_bytes(bits, n_bytes):
    """Convert bit array back to bytes."""
    bits = np.clip(np.round(bits), 0, 1).astype(np.uint8)
    # pad to multiple of 8
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()[:n_bytes]

def simulate_jpeg_awgn(img_uint8, ebno_db, quality=75, k=512, n=1024):
    """
    JPEG compress -> LDPC encode -> AWGN -> LDPC decode -> JPEG decompress
    Returns reconstructed image as uint8
    """
    # JPEG encode
    jpeg_bytes = jpeg_encode(img_uint8, quality)
    n_bytes = len(jpeg_bytes)
    bits = bytes_to_bits(jpeg_bytes)

    # Pad to multiple of k
    pad = (k - len(bits) % k) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.float32)])
    bits = bits.reshape(-1, k)
    n_blocks = bits.shape[0]

    # LDPC encode + AWGN + LDPC decode (using Sionna)
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, hard_out=True, num_iter=6)
    mapper  = Mapper("qam", num_bits_per_symbol=4)
    demapper = Demapper("app", "qam", num_bits_per_symbol=4)
    awgn_ch  = AWGN()

    b = tf.constant(bits, dtype=tf.float32)
    b = tf.reshape(b, (n_blocks, k))
    c = encoder(b)
    x = mapper(c)
    coderate = k / n
    no = ebnodb2no(ebno_db, num_bits_per_symbol=4, coderate=coderate)
    y = awgn_ch([x, no])
    llr = demapper([y, no])
    b_hat = decoder(llr).numpy()

    # Recover bits
    rec_bits = b_hat.flatten()[:len(bytes_to_bits(jpeg_bytes))]

    # JPEG decode
    try:
        rec_bytes = bits_to_bytes(rec_bits, n_bytes)
        rec_img = jpeg_decode(rec_bytes)
        if rec_img.shape != (32, 32, 3):
            return None
        return rec_img
    except:
        return None


def main():
    os.makedirs('checkpoints/image-jscc/eval', exist_ok=True)

    # Load test set
    (_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
    n_samples = 500  # use subset for speed

    snr_list = [-5, 0, 5, 10, 15, 20, 25]
    results = []

    for snr in snr_list:
        psnr_vals, ssim_vals = [], []
        success = 0

        for i in range(n_samples):
            img = x_te[i]  # uint8
            rec = simulate_jpeg_awgn(img, snr)
            if rec is not None:
                img_f = img.astype(np.float32) / 255.0
                rec_f = rec.astype(np.float32) / 255.0
                mse = np.mean((img_f - rec_f) ** 2)
                if mse > 0:
                    p = 10 * np.log10(1.0 / mse)
                else:
                    p = 100.0
                psnr_vals.append(p)
                # SSIM approximation
                ssim_vals.append(float(tf.image.ssim(
                    tf.constant(img_f[None]), tf.constant(rec_f[None]), max_val=1.0)))
                success += 1

        p_mean = float(np.mean(psnr_vals)) if psnr_vals else 0.0
        s_mean = float(np.mean(ssim_vals)) if ssim_vals else 0.0
        results.append({'ebno_db': snr, 'psnr': p_mean, 'ssim': s_mean, 'success_rate': success/n_samples})
        print(f"SNR={snr:4d} dB  PSNR={p_mean:.2f} dB  SSIM={s_mean:.4f}  Success={success}/{n_samples}")

    with open('checkpoints/image-jscc/eval/jpeg_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: checkpoints/image-jscc/eval/jpeg_baseline.json")


if __name__ == '__main__':
    main()

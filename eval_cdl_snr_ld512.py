import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import numpy as np, json, sys
from train_reconstruction import ReconstructionModel, psnr, ssim_metric

CDL_WEIGHTS  = sys.argv[1]
AWGN_WEIGHTS = sys.argv[2]
LATENT_DIM   = 512

dummy = tf.zeros((2, 32, 32, 3))
(_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
x_te = x_te.astype('float32') / 255.0
test_ds = tf.data.Dataset.from_tensor_slices(x_te).batch(32).prefetch(tf.data.AUTOTUNE)

# 加载 AWGN 预训练权重
m_awgn = ReconstructionModel(latent_dim=LATENT_DIM, bypass=False, channel_type='analog')
m_awgn(dummy, training=False)
m_awgn.load_weights(AWGN_WEIGHTS)

# 建 CDL 模型，复制 encoder/decoder，再加载 CDL 微调权重
m_cdl = ReconstructionModel(
    latent_dim=LATENT_DIM, bypass=False, channel_type='CDL',
    cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
    channel_num_tx_ant=2, channel_num_rx_ant=2,
    num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
)
m_cdl(dummy, training=False)
m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
m_cdl.load_weights(CDL_WEIGHTS)
print("CDL ld=512 模型加载完成")

results = []
for snr in [-5, 0, 5, 10, 15, 20, 25]:
    p_m = tf.keras.metrics.Mean()
    s_m = tf.keras.metrics.Mean()
    for imgs in test_ds:
        x_hat = tf.clip_by_value(m_cdl(imgs, ebno_db=float(snr), training=False), 0, 1)
        p_m.update_state(psnr(imgs, x_hat))
        s_m.update_state(ssim_metric(imgs, x_hat))
    p, s = float(p_m.result()), float(s_m.result())
    results.append({'ebno_db': snr, 'psnr': p, 'ssim': s})
    print(f"SNR={snr:4d} dB  PSNR={p:.2f}  SSIM={s:.4f}")

os.makedirs('checkpoints/image-jscc/eval', exist_ok=True)
with open('checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: snr_recon_cdl_ld512.json")

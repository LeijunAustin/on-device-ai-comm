import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import numpy as np, json
from train_reconstruction import ReconstructionModel, psnr, ssim_metric

CDL_WEIGHTS = 'checkpoints/image-jscc/recon_cdl_finetune_2026-03-24_08-35-49/best_psnr17.09.weights.h5'
AWGN_WEIGHTS = 'checkpoints/image-jscc/recon_ld128_2026-03-21_16-13-24/best_psnr21.55.weights.h5'

dummy = tf.zeros((2, 32, 32, 3))
(_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
x_te = x_te.astype('float32') / 255.0
test_ds = tf.data.Dataset.from_tensor_slices(x_te).batch(32).prefetch(tf.data.AUTOTUNE)

# 加载 CDL 模型
m_awgn = ReconstructionModel(latent_dim=128, bypass=False, channel_type='analog')
m_awgn(dummy, training=False)
m_awgn.load_weights(AWGN_WEIGHTS)

m_cdl = ReconstructionModel(
    latent_dim=128, bypass=False, channel_type='CDL',
    cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
    channel_num_tx_ant=2, channel_num_rx_ant=2,
    num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
)
m_cdl(dummy, training=False)
m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
m_cdl.load_weights(CDL_WEIGHTS)
print("CDL 模型加载完成")

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
with open('checkpoints/image-jscc/eval/snr_recon_cdl.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: checkpoints/image-jscc/eval/snr_recon_cdl.json")

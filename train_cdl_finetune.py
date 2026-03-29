import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import numpy as np, json
from datetime import datetime
from train_reconstruction import (ReconstructionModel, load_cifar10,
                                   reconstruction_loss, psnr, ssim_metric)

def main():
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out = f'checkpoints/image-jscc/recon_cdl_finetune_{ts}'
    os.makedirs(out, exist_ok=True)

    train_ds, test_ds = load_cifar10(batch_size=32)
    dummy = tf.zeros((2, 32, 32, 3))

    # 加载 AWGN 预训练权重
    m_awgn = ReconstructionModel(latent_dim=128, bypass=False, channel_type='analog')
    m_awgn(dummy, training=False)
    m_awgn.load_weights('checkpoints/image-jscc/recon_ld128_2026-03-21_16-13-24/best_psnr21.55.weights.h5')

    # 建 CDL 模型
    m_cdl = ReconstructionModel(
        latent_dim=128, bypass=False, channel_type='CDL',
        cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
        channel_num_tx_ant=2, channel_num_rx_ant=2,
        num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
    )
    m_cdl(dummy, training=False)
    m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
    m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
    print("预训练权重加载完成")

    opt = tf.keras.optimizers.Adam(1e-4)
    best = 0.0
    history = []

    for epoch in range(20):
        tr_loss = tf.keras.metrics.Mean()
        tr_psnr = tf.keras.metrics.Mean()

        for imgs in train_ds:
            with tf.GradientTape() as tape:
                x_hat = m_cdl(imgs, training=True)
                x_hat = tf.clip_by_value(x_hat, 0, 1)
                loss = reconstruction_loss(imgs, x_hat)
            grads = tape.gradient(loss, m_cdl.trainable_variables)
            opt.apply_gradients(zip(grads, m_cdl.trainable_variables))
            tr_loss.update_state(loss)
            tr_psnr.update_state(psnr(imgs, x_hat))

        te_psnr = tf.keras.metrics.Mean()
        te_ssim = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = tf.clip_by_value(m_cdl(imgs, training=False), 0, 1)
            te_psnr.update_state(psnr(imgs, x_hat))
            te_ssim.update_state(ssim_metric(imgs, x_hat))

        res = {'epoch': epoch+1,
               'train_psnr': float(tr_psnr.result()),
               'test_psnr':  float(te_psnr.result()),
               'test_ssim':  float(te_ssim.result())}
        history.append(res)
        print(f"Ep{epoch+1:3d}/20  TrainPSNR:{res['train_psnr']:.2f}  "
              f"TestPSNR:{res['test_psnr']:.2f}  SSIM:{res['test_ssim']:.4f}")

        if res['test_psnr'] > best:
            best = res['test_psnr']
            m_cdl.save_weights(f'{out}/best_psnr{best:.2f}.weights.h5')
            print(f"  ✅ Best: {best:.2f} dB")

        with open(f'{out}/history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best CDL PSNR: {best:.2f} dB")

if __name__ == '__main__':
    main()

"""
SNR sweep evaluation for reconstruction model
Generates PSNR/SSIM vs SNR curve
"""
import os, json, argparse
import numpy as np
import tensorflow as tf
from train_reconstruction import ReconstructionModel


def load_test(batch_size=64):
    (_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
    x_te = x_te.astype('float32') / 255.0
    return tf.data.Dataset.from_tensor_slices(x_te).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_ds = load_test(args.batch_size)

    model = ReconstructionModel(
        latent_dim=args.latent_dim,
        ebno_db_min=0, ebno_db_max=20,
        bypass=False,
    )
    dummy = tf.zeros((2, 32, 32, 3))
    model(dummy, training=False)
    model.load_weights(args.weights)
    print(f"Loaded: {args.weights}")

    # SNR sweep
    results = []
    snr_list = list(range(args.snr_min, args.snr_max + 1, args.snr_step))

    for snr in snr_list:
        psnr_m = tf.keras.metrics.Mean()
        ssim_m = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = model(imgs, ebno_db=float(snr), training=False)
            x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)
            psnr_m.update_state(tf.image.psnr(imgs, x_hat, max_val=1.0))
            ssim_m.update_state(tf.image.ssim(imgs, x_hat, max_val=1.0))
        p = float(psnr_m.result())
        s = float(ssim_m.result())
        results.append({'ebno_db': snr, 'psnr': p, 'ssim': s})
        print(f"SNR={snr:4d} dB  PSNR={p:.2f} dB  SSIM={s:.4f}")

    # 也评估无信道上界
    psnr_m = tf.keras.metrics.Mean()
    ssim_m = tf.keras.metrics.Mean()
    for imgs in test_ds:
        bypass_model = ReconstructionModel(latent_dim=args.latent_dim, bypass=True)
        bypass_model(dummy, training=False)
        bypass_model.encoder.set_weights(model.encoder.get_weights())
        bypass_model.decoder.set_weights(model.decoder.get_weights())
        x_hat = bypass_model(imgs, training=False)
        x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)
        psnr_m.update_state(tf.image.psnr(imgs, x_hat, max_val=1.0))
        ssim_m.update_state(tf.image.ssim(imgs, x_hat, max_val=1.0))
        break  # 只需建一次模型
    # 重新算
    bypass_model = ReconstructionModel(latent_dim=args.latent_dim, bypass=True)
    bypass_model(dummy, training=False)
    bypass_model.encoder.set_weights(model.encoder.get_weights())
    bypass_model.decoder.set_weights(model.decoder.get_weights())
    psnr_m = tf.keras.metrics.Mean()
    ssim_m = tf.keras.metrics.Mean()
    for imgs in test_ds:
        x_hat = bypass_model(imgs, training=False)
        x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)
        psnr_m.update_state(tf.image.psnr(imgs, x_hat, max_val=1.0))
        ssim_m.update_state(tf.image.ssim(imgs, x_hat, max_val=1.0))
    print(f"\nBypass (no channel): PSNR={psnr_m.result():.2f} dB  SSIM={ssim_m.result():.4f}")

    out_path = os.path.join(args.output_dir, f'snr_recon_ld{args.latent_dim}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',     required=True)
    p.add_argument('--latent-dim',  type=int, default=128)
    p.add_argument('--snr-min',     type=int, default=-5)
    p.add_argument('--snr-max',     type=int, default=25)
    p.add_argument('--snr-step',    type=int, default=5)
    p.add_argument('--batch-size',  type=int, default=64)
    p.add_argument('--output-dir',  default='checkpoints/image-jscc/eval')
    return p.parse_args()

if __name__ == '__main__':
    main(parse_args())

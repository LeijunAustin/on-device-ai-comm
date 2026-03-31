"""
eval_reconstruction.py — AWGN 信道 SNR sweep 评估脚本

用法：
  python eval_reconstruction.py --weights <path> --latent-dim 128
  python eval_reconstruction.py --weights <path> --latent-dim 512 --snr-min -5 --snr-max 25

输出：
  checkpoints/image-jscc/eval/snr_recon_ld{N}.json
"""

import os, json, argparse
import tensorflow as tf
from train_reconstruction import ReconstructionModel


def load_test(batch_size=64):
    (_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
    x_te = x_te.astype('float32') / 255.0
    return tf.data.Dataset.from_tensor_slices(x_te).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_ds = load_test(args.batch_size)
    dummy   = tf.zeros((2, 32, 32, 3))

    # ── 加载 AWGN 模型 ────────────────────────────────────────────
    model = ReconstructionModel(
        latent_dim=args.latent_dim,
        ebno_db_min=0, ebno_db_max=20,
        bypass=False,
    )
    model(dummy, training=False)
    model.load_weights(args.weights)
    print(f"Loaded: {args.weights}\n")

    # ── SNR sweep ─────────────────────────────────────────────────
    results = []
    snr_list = list(range(args.snr_min, args.snr_max + 1, args.snr_step))

    for snr in snr_list:
        psnr_m = tf.keras.metrics.Mean()
        ssim_m = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = tf.clip_by_value(
                model(imgs, ebno_db=float(snr), training=False), 0.0, 1.0)
            psnr_m.update_state(tf.image.psnr(imgs, x_hat, max_val=1.0))
            ssim_m.update_state(tf.image.ssim(imgs, x_hat, max_val=1.0))
        p = float(psnr_m.result())
        s = float(ssim_m.result())
        results.append({'ebno_db': snr, 'psnr': p, 'ssim': s})
        print(f"SNR={snr:4d} dB  PSNR={p:.2f} dB  SSIM={s:.4f}")

    # ── 无信道上界（bypass 模式）──────────────────────────────────
    # bypass 模型结构完全一样，只是信道层被跳过，
    # encoder/decoder 权重直接从已训练的 AWGN 模型复制过来，
    # 这样评估的是"如果信道完全无噪声，模型能达到多高的 PSNR"。
    bypass_model = ReconstructionModel(latent_dim=args.latent_dim, bypass=True)
    bypass_model(dummy, training=False)
    bypass_model.encoder.set_weights(model.encoder.get_weights())
    bypass_model.decoder.set_weights(model.decoder.get_weights())

    psnr_m = tf.keras.metrics.Mean()
    ssim_m = tf.keras.metrics.Mean()
    for imgs in test_ds:
        x_hat = tf.clip_by_value(bypass_model(imgs, training=False), 0.0, 1.0)
        psnr_m.update_state(tf.image.psnr(imgs, x_hat, max_val=1.0))
        ssim_m.update_state(tf.image.ssim(imgs, x_hat, max_val=1.0))
    print(f"\nBypass (no channel): PSNR={psnr_m.result():.2f} dB  "
          f"SSIM={ssim_m.result():.4f}")

    # ── 保存结果 ──────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f'snr_recon_ld{args.latent_dim}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description='AWGN 信道 SNR sweep 评估')
    p.add_argument('--weights',    required=True,  help='模型权重路径 (.weights.h5)')
    p.add_argument('--latent-dim', type=int, default=128, choices=[128, 256, 512])
    p.add_argument('--snr-min',    type=int, default=-5)
    p.add_argument('--snr-max',    type=int, default=25)
    p.add_argument('--snr-step',   type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output-dir', default='checkpoints/image-jscc/eval')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())

"""
重建图像可视化脚本
用法：
    conda activate on-device-ai-comm
    python visualize_reconstruction.py \
        --weights checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5 \
        --latent-dim 512

输出：
    checkpoints/image-jscc/eval/reconstruction_visual.png
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')

from train_reconstruction import ReconstructionModel, psnr as compute_psnr

SNR_POINTS = [-5, 0, 5, 10, 20]
NUM_IMAGES  = 6   # 每行展示几张图


def pick_diverse_images(x_test, n=NUM_IMAGES, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=n, replace=False)
    return x_test[idx]


def render_grid(model, images, snr_points, out_path):
    """
    行 = SNR 等级（从上到下：原图、SNR 从低到高）
    列 = 不同图片样本
    """
    n_snr  = len(snr_points)
    n_imgs = len(images)
    n_rows  = n_snr + 1   # +1 for original
    n_cols  = n_imgs

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.6, n_rows * 1.8),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
    )

    # 计算每张图每个 SNR 的 PSNR，用于标题
    psnr_grid = np.zeros((n_snr, n_imgs))
    recon_grid = np.zeros((n_snr, n_imgs, 32, 32, 3))

    for si, snr in enumerate(snr_points):
        batch = tf.constant(images, dtype=tf.float32)
        x_hat = tf.clip_by_value(
            model(batch, ebno_db=float(snr), training=False), 0.0, 1.0
        ).numpy()
        recon_grid[si] = x_hat
        for ii in range(n_imgs):
            psnr_val = float(compute_psnr(
                tf.expand_dims(images[ii], 0),
                tf.expand_dims(x_hat[ii], 0)
            ))
            psnr_grid[si, ii] = psnr_val

    # 第 0 行：原图
    for ii in range(n_imgs):
        ax = axes[0, ii]
        ax.imshow(images[ii])
        ax.set_xticks([]); ax.set_yticks([])
        if ii == 0:
            ax.set_ylabel('Original', fontsize=8, fontweight='bold')

    # 后续行：各 SNR 下的重建
    for si, snr in enumerate(snr_points):
        for ii in range(n_imgs):
            ax = axes[si + 1, ii]
            ax.imshow(recon_grid[si, ii])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{psnr_grid[si, ii]:.1f} dB', fontsize=6, pad=2)
            if ii == 0:
                ax.set_ylabel(f'SNR={snr:+d} dB', fontsize=8, fontweight='bold')

    fig.suptitle(
        'Image Reconstruction Quality vs SNR — CNN JSCC (ld=512, AWGN)\n'
        'Columns: different test images  |  Rows: original + reconstructed at various SNR\n'
        'Numbers: PSNR (dB)',
        fontsize=10, fontweight='bold', y=1.01
    )

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',     required=True)
    parser.add_argument('--latent-dim',  type=int, default=512)
    parser.add_argument('--num-images',  type=int, default=NUM_IMAGES)
    parser.add_argument('--output-dir',  default='checkpoints/image-jscc/eval')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    (_, _), (x_test_raw, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test_raw.astype('float32') / 255.0

    dummy = tf.zeros((2, 32, 32, 3))
    model = ReconstructionModel(
        latent_dim=args.latent_dim, bypass=False, channel_type='analog')
    model(dummy, training=False)
    model.load_weights(args.weights)
    print(f"权重加载: {args.weights}")

    images = pick_diverse_images(x_test, n=args.num_images, seed=args.seed)

    out_path = os.path.join(args.output_dir, 'reconstruction_visual.png')
    render_grid(model, images, SNR_POINTS, out_path)

    # 打印数值汇总
    print(f"\n{'SNR':>8} | {'平均 PSNR':>10}")
    print('-' * 22)
    for si, snr in enumerate(SNR_POINTS):
        batch = tf.constant(x_test[:500], dtype=tf.float32)
        psnrs = []
        for b in tf.data.Dataset.from_tensor_slices(batch).batch(64):
            x_hat = tf.clip_by_value(model(b, ebno_db=float(snr), training=False), 0, 1)
            psnrs.append(float(compute_psnr(b, x_hat)))
        print(f"{snr:>+8d} | {np.mean(psnrs):>10.2f} dB")


if __name__ == '__main__':
    main()

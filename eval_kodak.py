"""
Kodak 数据集评估脚本
将 Kodak 24 张图像裁为不重叠的 32×32 patch，用已有权重评估 PSNR/SSIM。

用法：
    python eval_kodak.py \
        --weights checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5 \
        --latent-dim 512
"""

import argparse, os, json, urllib.request, glob
import numpy as np
import tensorflow as tf

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')

from train_reconstruction import ReconstructionModel, psnr as compute_psnr, ssim_metric

KODAK_DIR   = os.path.expanduser('~/.keras/datasets/kodak')
KODAK_BASE  = 'http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png'
NUM_IMAGES  = 24
SNR_POINTS  = [-5, 0, 5, 10, 15, 20, 25]
PATCH_SIZE  = 32


def download_kodak():
    os.makedirs(KODAK_DIR, exist_ok=True)
    for i in range(1, NUM_IMAGES + 1):
        dst = os.path.join(KODAK_DIR, f'kodim{i:02d}.png')
        if not os.path.exists(dst):
            url = KODAK_BASE.format(i)
            print(f'Downloading {url}')
            urllib.request.urlretrieve(url, dst)
    paths = sorted(glob.glob(os.path.join(KODAK_DIR, '*.png')))
    print(f'Kodak: {len(paths)} images found in {KODAK_DIR}')
    return paths


def extract_patches(img_uint8, patch_size=PATCH_SIZE):
    """Crop non-overlapping patch_size×patch_size patches, discard remainder."""
    h, w = img_uint8.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append(img_uint8[y:y+patch_size, x:x+patch_size])
    return patches


def load_kodak_patches(paths):
    from PIL import Image
    patches = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
        patches.extend(extract_patches(img))
    patches = np.array(patches, dtype=np.float32) / 255.0
    print(f'Total patches: {len(patches)} (from {len(paths)} Kodak images)')
    return patches


def find_weights(latent_dim, weights_override=None):
    if weights_override:
        return weights_override
    candidates = sorted(
        glob.glob(f'checkpoints/image-jscc/recon_ld{latent_dim}_*/best_psnr*.weights.h5'),
        key=lambda p: float(os.path.basename(p).replace('best_psnr','').replace('.weights.h5',''))
    )
    if not candidates:
        raise FileNotFoundError(f'No AWGN weights found for ld={latent_dim}')
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    default=None)
    parser.add_argument('--latent-dim', type=int, default=512)
    parser.add_argument('--output-dir', default='checkpoints/image-jscc/eval')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = find_weights(args.latent_dim, args.weights)
    print(f'Weights: {weights_path}')

    # 构建模型
    dummy = tf.zeros((2, PATCH_SIZE, PATCH_SIZE, 3))
    model = ReconstructionModel(latent_dim=args.latent_dim, bypass=False, channel_type='analog')
    model(dummy, training=False)
    model.load_weights(weights_path)

    # 加载数据
    paths   = download_kodak()
    patches = load_kodak_patches(paths)

    results = []
    for snr in SNR_POINTS:
        psnrs, ssims = [], []
        ds = tf.data.Dataset.from_tensor_slices(patches).batch(64)
        for batch in ds:
            x_hat = tf.clip_by_value(model(batch, ebno_db=float(snr), training=False), 0.0, 1.0)
            psnrs.append(float(compute_psnr(batch, x_hat)))
            ssims.append(float(ssim_metric(batch, x_hat)))
        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))
        results.append({'ebno_db': snr, 'psnr': avg_psnr, 'ssim': avg_ssim})
        print(f'  SNR={snr:+3d} dB  PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}')

    out = os.path.join(args.output_dir, f'kodak_ld{args.latent_dim}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}')

    # 与 CIFAR-10 对比
    cifar_path = os.path.join(args.output_dir, f'snr_recon_ld{args.latent_dim}.json')
    if os.path.exists(cifar_path):
        with open(cifar_path) as f:
            cifar = json.load(f)
        print(f"\n{'SNR':>6} {'Kodak':>8} {'CIFAR-10':>10} {'Diff':>6}")
        print('-' * 36)
        for kr, cr in zip(results, cifar):
            diff = kr['psnr'] - cr['psnr']
            print(f"{kr['ebno_db']:>+6d}  {kr['psnr']:>8.2f}  {cr['psnr']:>10.2f}  {diff:>+6.2f}")


if __name__ == '__main__':
    main()

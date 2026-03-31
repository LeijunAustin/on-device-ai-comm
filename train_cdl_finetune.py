import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import glob, argparse, json
from datetime import datetime
from train_reconstruction import (ReconstructionModel, load_cifar10,
                                   reconstruction_loss, psnr, ssim_metric)

def find_best_awgn_weights(latent_dim):
    """
    自动找到对应 latent_dim 的最佳 AWGN 预训练权重。
    好处：不用在代码里硬编码路径，每次重新训练后自动用最新最好的权重。
    规则：glob 找到所有候选文件，按文件名里的 PSNR 数值取最高的那个。
    """
    pattern = f"checkpoints/image-jscc/recon_ld{latent_dim}_*/best_psnr*.weights.h5"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"找不到 latent_dim={latent_dim} 的 AWGN 权重，"
            f"请先运行: python train_reconstruction.py --latent-dim {latent_dim}"
        )
    # 从文件名 best_psnr23.92.weights.h5 中提取数值 23.92，取最大的
    def extract_psnr(path):
        name = os.path.basename(path)              # best_psnr23.92.weights.h5
        return float(name.replace('best_psnr', '').replace('.weights.h5', ''))

    best_path = max(candidates, key=extract_psnr)
    print(f"自动找到 AWGN 权重 (ld={latent_dim}): {best_path}")
    return best_path


def main(args):
    ts  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 输出目录名里包含 latent_dim，方便区分不同配置的结果
    out = f'checkpoints/image-jscc/recon_cdl_finetune_ld{args.latent_dim}_{ts}'
    os.makedirs(out, exist_ok=True)

    train_ds, test_ds = load_cifar10(batch_size=32)
    dummy = tf.zeros((2, 32, 32, 3))

    # ── 第一步：加载 AWGN 预训练权重 ──────────────────────────────
    # 自动查找，不依赖硬编码路径
    awgn_weights = find_best_awgn_weights(args.latent_dim)
    m_awgn = ReconstructionModel(latent_dim=args.latent_dim, bypass=False, channel_type='analog')
    m_awgn(dummy, training=False)
    m_awgn.load_weights(awgn_weights)

    # ── 第二步：建 CDL 模型，把 AWGN 的 encoder/decoder 权重复制过来 ──
    # 这是两阶段训练的核心：CDL 模型从已经学好的特征表示出发微调，
    # 而不是从随机初始化开始，所以收敛快且性能好。
    m_cdl = ReconstructionModel(
        latent_dim=args.latent_dim, bypass=False, channel_type='CDL',
        cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
        channel_num_tx_ant=2, channel_num_rx_ant=2,
        num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
    )
    m_cdl(dummy, training=False)
    m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
    m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
    print(f"预训练权重加载完成，开始 CDL 微调 (ld={args.latent_dim})...")

    # ── 第三步：微调训练 ───────────────────────────────────────────
    opt     = tf.keras.optimizers.Adam(args.lr)
    best    = 0.0
    history = []

    for epoch in range(args.epochs):
        tr_psnr = tf.keras.metrics.Mean()

        for imgs in train_ds:
            with tf.GradientTape() as tape:
                x_hat = tf.clip_by_value(m_cdl(imgs, training=True), 0, 1)
                loss  = reconstruction_loss(imgs, x_hat)
            grads = tape.gradient(loss, m_cdl.trainable_variables)
            opt.apply_gradients(zip(grads, m_cdl.trainable_variables))
            tr_psnr.update_state(psnr(imgs, x_hat))

        te_psnr = tf.keras.metrics.Mean()
        te_ssim = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = tf.clip_by_value(m_cdl(imgs, training=False), 0, 1)
            te_psnr.update_state(psnr(imgs, x_hat))
            te_ssim.update_state(ssim_metric(imgs, x_hat))

        res = {'epoch':      epoch + 1,
               'train_psnr': float(tr_psnr.result()),
               'test_psnr':  float(te_psnr.result()),
               'test_ssim':  float(te_ssim.result())}
        history.append(res)
        print(f"Ep{epoch+1:3d}/{args.epochs}  "
              f"TrainPSNR:{res['train_psnr']:.2f}  "
              f"TestPSNR:{res['test_psnr']:.2f}  "
              f"SSIM:{res['test_ssim']:.4f}")

        if res['test_psnr'] > best:
            best = res['test_psnr']
            m_cdl.save_weights(f'{out}/best_psnr{best:.2f}.weights.h5')
            print(f"  ✅ Best: {best:.2f} dB")

        # history.json 每个 epoch 都更新，训练中断也不会丢数据
        with open(f'{out}/history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best CDL PSNR (ld={args.latent_dim}): {best:.2f} dB → {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDL-A 信道微调脚本（两阶段训练第二阶段）',
        epilog='示例: python train_cdl_finetune.py --latent-dim 512'
    )
    parser.add_argument('--latent-dim', type=int, required=True,
                        choices=[128, 256, 512],
                        help='latent 维度，必须和已训练的 AWGN 模型一致')
    parser.add_argument('--epochs', type=int, default=20,
                        help='微调 epoch 数（默认 20）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率（默认 1e-4）')
    args = parser.parse_args()
    main(args)

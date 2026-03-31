"""
eval_cdl_snr.py — CDL 信道 SNR sweep 评估脚本

用法：
  python eval_cdl_snr.py --latent-dim 128
  python eval_cdl_snr.py --latent-dim 512
  python eval_cdl_snr.py --latent-dim 256

工作原理：
  自动找到对应 latent_dim 的最佳 AWGN 权重和 CDL 微调权重，
  在 SNR = [-5, 0, 5, 10, 15, 20, 25] dB 下评估 PSNR 和 SSIM，
  结果保存到 checkpoints/image-jscc/eval/snr_recon_cdl_ld{N}.json
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import glob, argparse, json
from train_reconstruction import ReconstructionModel, psnr, ssim_metric


def find_best_weights(pattern, label):
    """
    给定 glob 模式，自动找到 PSNR 数值最高的权重文件。
    出错时给出清晰的提示，而不是让用户面对 FileNotFoundError。
    """
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"找不到{label}权重: {pattern}\n"
            f"请先运行对应的训练脚本。"
        )
    def extract_psnr(path):
        name = os.path.basename(path)              # best_psnr23.92.weights.h5
        return float(name.replace('best_psnr', '').replace('.weights.h5', ''))
    best = max(candidates, key=extract_psnr)
    print(f"找到{label}权重: {best}")
    return best


def main(args):
    # ── 自动查找权重，不依赖硬编码路径 ────────────────────────────
    awgn_weights = find_best_weights(
        f"checkpoints/image-jscc/recon_ld{args.latent_dim}_*/best_psnr*.weights.h5",
        f"AWGN ld={args.latent_dim}"
    )
    cdl_weights = find_best_weights(
        f"checkpoints/image-jscc/recon_cdl_finetune_ld{args.latent_dim}_*/best_psnr*.weights.h5",
        f"CDL ld={args.latent_dim}"
    )

    # ── 加载测试集 ────────────────────────────────────────────────
    dummy = tf.zeros((2, 32, 32, 3))
    (_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
    x_te = x_te.astype('float32') / 255.0
    test_ds = (tf.data.Dataset.from_tensor_slices(x_te)
               .batch(32).prefetch(tf.data.AUTOTUNE))

    # ── 加载 AWGN 模型（用于复制 encoder/decoder 权重）────────────
    # CDL 模型的 encoder/decoder 结构和 AWGN 完全一致，
    # 需要先把 AWGN 权重复制过来，再叠加 CDL 微调权重。
    m_awgn = ReconstructionModel(
        latent_dim=args.latent_dim, bypass=False, channel_type='analog')
    m_awgn(dummy, training=False)
    m_awgn.load_weights(awgn_weights)

    # ── 建 CDL 模型，复制权重，加载微调权重 ──────────────────────
    m_cdl = ReconstructionModel(
        latent_dim=args.latent_dim, bypass=False, channel_type='CDL',
        cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
        channel_num_tx_ant=2, channel_num_rx_ant=2,
        num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
    )
    m_cdl(dummy, training=False)
    m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
    m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
    m_cdl.load_weights(cdl_weights)
    print(f"CDL ld={args.latent_dim} 模型加载完成\n")

    # ── SNR sweep ─────────────────────────────────────────────────
    results = []
    for snr in [-5, 0, 5, 10, 15, 20, 25]:
        p_m = tf.keras.metrics.Mean()
        s_m = tf.keras.metrics.Mean()
        for imgs in test_ds:
            x_hat = tf.clip_by_value(
                m_cdl(imgs, ebno_db=float(snr), training=False), 0, 1)
            p_m.update_state(psnr(imgs, x_hat))
            s_m.update_state(ssim_metric(imgs, x_hat))
        p, s = float(p_m.result()), float(s_m.result())
        results.append({'ebno_db': snr, 'psnr': p, 'ssim': s})
        print(f"SNR={snr:4d} dB  PSNR={p:.2f} dB  SSIM={s:.4f}")

    # ── 保存结果 ──────────────────────────────────────────────────
    # ld=128 的历史文件名是 snr_recon_cdl.json（没有 ld 编号），
    # 为了向后兼容 plot_results_final.py 里的路径，ld=128 保持原名。
    os.makedirs('checkpoints/image-jscc/eval', exist_ok=True)
    if args.latent_dim == 128:
        out_path = 'checkpoints/image-jscc/eval/snr_recon_cdl.json'
    else:
        out_path = f'checkpoints/image-jscc/eval/snr_recon_cdl_ld{args.latent_dim}.json'

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDL-A 信道 SNR sweep 评估',
        epilog='示例: python eval_cdl_snr.py --latent-dim 512'
    )
    parser.add_argument('--latent-dim', type=int, required=True,
                        choices=[128, 256, 512],
                        help='latent 维度，必须和已训练的 CDL 模型一致')
    main(parser.parse_args())

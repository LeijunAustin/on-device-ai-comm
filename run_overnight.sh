#!/bin/bash
set -e
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

LOG_DIR="checkpoints/image-jscc/overnight_logs"
mkdir -p $LOG_DIR
echo "====== 夜间实验开始: $(date) ======" | tee $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验1：ld=256 bypass（无信道上界）
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [1/6] ld=256 bypass 50ep: $(date)" | tee -a $LOG_DIR/master.log
python train_reconstruction.py \
    --latent-dim 256 --bypass \
    --batch-size 64 --epochs 50 --lr 1e-3 \
    --output-dir checkpoints/image-jscc \
    2>&1 | tee $LOG_DIR/exp1_ld256_bypass.txt
echo "实验1完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验2：ld=256 AWGN 信道训练
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [2/6] ld=256 AWGN 50ep: $(date)" | tee -a $LOG_DIR/master.log
python train_reconstruction.py \
    --latent-dim 256 \
    --ebno-db-min 0 --ebno-db-max 20 \
    --batch-size 64 --epochs 50 --lr 1e-3 \
    --output-dir checkpoints/image-jscc \
    2>&1 | tee $LOG_DIR/exp2_ld256_awgn.txt
echo "实验2完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验3：ld=256 CDL 微调
# 自动找实验2生成的最新权重
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [3/6] ld=256 CDL 微调 20ep: $(date)" | tee -a $LOG_DIR/master.log

# 找实验2最新的权重（按时间排序取最新）
LD256_AWGN_WEIGHTS=$(ls -t checkpoints/image-jscc/recon_ld256_*/best_psnr*.weights.h5 2>/dev/null | head -1)
echo "ld=256 AWGN 权重: $LD256_AWGN_WEIGHTS" | tee -a $LOG_DIR/master.log

python3 << PYEOF
import os, json
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import numpy as np
from datetime import datetime
from train_reconstruction import (ReconstructionModel, load_cifar10,
                                   reconstruction_loss, psnr, ssim_metric)

AWGN_WEIGHTS = "$LD256_AWGN_WEIGHTS"
LATENT_DIM = 256
ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out = f'checkpoints/image-jscc/recon_cdl_finetune_ld256_{ts}'
os.makedirs(out, exist_ok=True)

train_ds, test_ds = load_cifar10(batch_size=32)
dummy = tf.zeros((2, 32, 32, 3))

m_awgn = ReconstructionModel(latent_dim=LATENT_DIM, bypass=False, channel_type='analog')
m_awgn(dummy, training=False)
m_awgn.load_weights(AWGN_WEIGHTS)

m_cdl = ReconstructionModel(
    latent_dim=LATENT_DIM, bypass=False, channel_type='CDL',
    cdl_model='A', fec_type='LDPC5G', fec_num_iter=6,
    channel_num_tx_ant=2, channel_num_rx_ant=2,
    num_bits_per_symbol=4, ebno_db_min=5, ebno_db_max=15,
)
m_cdl(dummy, training=False)
m_cdl.encoder.set_weights(m_awgn.encoder.get_weights())
m_cdl.decoder.set_weights(m_awgn.decoder.get_weights())
print(f"ld=256 CDL 微调开始，权重来自: {AWGN_WEIGHTS}")

opt = tf.keras.optimizers.Adam(1e-4)
best = 0.0
history = []
for epoch in range(20):
    tr_psnr = tf.keras.metrics.Mean()
    for imgs in train_ds:
        with tf.GradientTape() as tape:
            x_hat = tf.clip_by_value(m_cdl(imgs, training=True), 0, 1)
            loss = reconstruction_loss(imgs, x_hat)
        grads = tape.gradient(loss, m_cdl.trainable_variables)
        opt.apply_gradients(zip(grads, m_cdl.trainable_variables))
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

print(f"ld=256 CDL 微调完成，Best PSNR: {best:.2f} dB -> {out}")
PYEOF
echo "实验3完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验4：ld=128 继续训练到 100 epoch
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [4/6] ld=128 AWGN 100ep: $(date)" | tee -a $LOG_DIR/master.log
python train_reconstruction.py \
    --latent-dim 128 \
    --ebno-db-min 0 --ebno-db-max 20 \
    --batch-size 64 --epochs 100 --lr 5e-4 \
    --output-dir checkpoints/image-jscc \
    2>&1 | tee $LOG_DIR/exp4_ld128_awgn_100ep.txt
echo "实验4完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验5：ld=512 继续训练到 100 epoch
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [5/6] ld=512 AWGN 100ep: $(date)" | tee -a $LOG_DIR/master.log
python train_reconstruction.py \
    --latent-dim 512 \
    --ebno-db-min 0 --ebno-db-max 20 \
    --batch-size 64 --epochs 100 --lr 5e-4 \
    --output-dir checkpoints/image-jscc \
    2>&1 | tee $LOG_DIR/exp5_ld512_awgn_100ep.txt
echo "实验5完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验6：ld=256 SNR sweep 评估
# 自动找 ld=256 AWGN 和 CDL 最新权重
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [6/6] ld=256 SNR sweep: $(date)" | tee -a $LOG_DIR/master.log

LD256_AWGN_WEIGHTS=$(ls -t checkpoints/image-jscc/recon_ld256_*/best_psnr*.weights.h5 2>/dev/null | head -1)
LD256_CDL_WEIGHTS=$(ls -t checkpoints/image-jscc/recon_cdl_finetune_ld256_*/best_psnr*.weights.h5 2>/dev/null | head -1)
echo "AWGN 权重: $LD256_AWGN_WEIGHTS" | tee -a $LOG_DIR/master.log
echo "CDL  权重: $LD256_CDL_WEIGHTS"  | tee -a $LOG_DIR/master.log

python eval_reconstruction.py \
    --weights "$LD256_AWGN_WEIGHTS" \
    --latent-dim 256 \
    --snr-min -5 --snr-max 25 --snr-step 5 \
    --output-dir checkpoints/image-jscc/eval \
    2>&1 | tee $LOG_DIR/exp6a_ld256_awgn_snr.txt

# CDL SNR sweep
python3 << PYEOF
import os, json
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import numpy as np
from train_reconstruction import ReconstructionModel, psnr, ssim_metric

CDL_WEIGHTS  = "$LD256_CDL_WEIGHTS"
AWGN_WEIGHTS = "$LD256_AWGN_WEIGHTS"
LATENT_DIM = 256

dummy = tf.zeros((2, 32, 32, 3))
(_, _), (x_te, _) = tf.keras.datasets.cifar10.load_data()
x_te = x_te.astype('float32') / 255.0
test_ds = tf.data.Dataset.from_tensor_slices(x_te).batch(32).prefetch(tf.data.AUTOTUNE)

m_awgn = ReconstructionModel(latent_dim=LATENT_DIM, bypass=False, channel_type='analog')
m_awgn(dummy, training=False)
m_awgn.load_weights(AWGN_WEIGHTS)

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
print("ld=256 CDL 模型加载完成")

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

with open('checkpoints/image-jscc/eval/snr_recon_cdl_ld256.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: snr_recon_cdl_ld256.json")
PYEOF
echo "实验6完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验7：Kodak 数据集评估（ld=128/256/512）
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [7/9] Kodak 评估: $(date)" | tee -a $LOG_DIR/master.log
for LD in 128 256 512; do
    python eval_kodak.py --latent-dim $LD \
        2>&1 | tee $LOG_DIR/kodak_ld${LD}.txt
done
echo "实验7完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验8：固定 CBR JPEG+LDPC 对比
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [8/9] 固定CBR JPEG评估: $(date)" | tee -a $LOG_DIR/master.log
python eval_jpeg_fixed_cbr.py \
    2>&1 | tee $LOG_DIR/jpeg_fixed_cbr.txt
echo "实验8完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 实验9：感知损失微调（ld=512，从 AWGN 权重）
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> [9/9] 感知损失微调: $(date)" | tee -a $LOG_DIR/master.log
LD512_AWGN_WEIGHTS=$(ls -t checkpoints/image-jscc/recon_ld512_2*/best_psnr*.weights.h5 2>/dev/null | grep -v perceptual | head -1)
echo "基础权重: $LD512_AWGN_WEIGHTS" | tee -a $LOG_DIR/master.log
python train_reconstruction.py \
    --latent-dim 512 --epochs 30 --lr 5e-5 \
    --perceptual-weight 0.1 \
    --finetune-from "$LD512_AWGN_WEIGHTS" \
    2>&1 | tee $LOG_DIR/perceptual_finetune.txt
echo "实验9完成: $(date)" | tee -a $LOG_DIR/master.log

# ─────────────────────────────────────────
# 最终：重新生成对比图
# ─────────────────────────────────────────
echo "" | tee -a $LOG_DIR/master.log
echo ">>> 生成最终图表: $(date)" | tee -a $LOG_DIR/master.log
python plot_results_final.py 2>&1 | tee $LOG_DIR/plot_final.txt

echo "" | tee -a $LOG_DIR/master.log
echo "====== 所有实验完成: $(date) ======" | tee -a $LOG_DIR/master.log

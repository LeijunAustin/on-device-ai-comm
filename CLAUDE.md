# Image Semantic Communication Project — 项目状态文档
> 最后更新：2026年3月25日
> 用途：在新对话中恢复项目上下文，让 Claude 能继续帮助推进项目

---

## 0. 如何使用这份文档

把这份文档的内容粘贴到新对话的开头，然后告诉 Claude："这是我的项目状态文档，请帮我继续推进。"

---

## 1. 项目背景

**题目：** Site-specific Semantic E2E PUSCH Encoding/Decoding for Image Transmission

**参考仓库：** https://github.com/abman23/on-device-ai-comm （已 fork 到 https://github.com/LeijunAustin/on-device-ai-comm）

**老师要求：**
1. 把原项目的语言语义通信（BART + 文字）改造成图像语义通信，保留 5G-NR PUSCH 物理层结构
2. 与传统图像传输技术（JPEG+LDPC）对比验证性能
3. 用 Sionna RT 射线追踪在 site-specific 场景下验证

**最终任务类型：** 图像重建（Image Reconstruction），指标为 PSNR / SSIM vs SNR

---

## 2. 环境配置
```
OS：Windows 11 + WSL2 Ubuntu 22.04
GPU：NVIDIA RTX 4070 Laptop，8GB 显存
CUDA：11.2
conda 环境：on-device-ai-comm
Python：3.9
TensorFlow：2.10.1
Sionna：0.11.0（本地，不含 RT 模块，RT 需要在 Google Colab 上运行）
```

**激活环境：**
```bash
conda activate on-device-ai-comm
cd ~/on-device-ai-comm
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export DRJIT_LIBLLVM_PATH=/usr/lib/llvm-14/lib/libLLVM-14.so
```

**重要注意事项：**
- CDL 信道必须设置 `TF_FORCE_GPU_ALLOW_GROWTH=true`，否则 cuSolverDN crash
- Sionna 0.11.0 不含 RT 模块，RT 实验需要在 Google Colab 上运行新版 Sionna

---

## 3. 项目文件结构

**原项目文件（未动或小改）：**
- `models/on_device_ai_comm.py`：原项目 BART 语言模型，未改动
- `models/vq_vae.py`：向量量化器，未改动
- `models/channels.py`：信道模块，**有两处修改**：
  1. 所有 `call` 函数加了 `ebno_db_override=None` 参数，用于 SNR sweep 时固定 SNR
  2. 去掉了 `@tf.function(jit_compile=True)`，改为 `@tf.function`，解决 WSL2 兼容性
  - **修改向后兼容，原项目可以正常运行**
- `train.py`、`scripts/train.sh`、`scripts/eval.sh`：原项目训练评估，未改动

**新建的图像语义通信文件：**
- `models/image_semantic_comm.py`：图像语义模型定义（CNN encoder/decoder，分类/重建 decoder）
- `train_reconstruction.py`：图像重建训练脚本，支持 analog/CDL 两种信道，两阶段训练
- `train_cdl_finetune.py`：ld=128 CDL 微调脚本
- `train_cdl_finetune_ld512.py`：ld=512 CDL 微调脚本
- `eval_reconstruction.py`：AWGN 信道 SNR sweep 评估
- `eval_cdl_snr.py`：CDL 信道 SNR sweep 评估（ld=128）
- `eval_cdl_snr_ld512.py`：CDL 信道 SNR sweep 评估（ld=512）
- `eval_jpeg_baseline.py`：JPEG+LDPC 传统 baseline 评估
- `plot_final_v5.py`：生成最终六条曲线对比图
- `run_overnight.sh`：夜间批处理实验脚本

---

## 4. 已完成实验结果

### 4.1 原项目复现（文字语义通信）
- 信道：CDL-A，SNR 5-15 dB
- BLEU = 0.935，rouge1 = 99.74%，BER = 0.128 @4dB
- 权重：`checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15/`

### 4.2 图像重建系统（主要贡献）

**训练策略：两阶段训练**
- Phase 1：用 AnalogAWGN（可微分）预训练 encoder + decoder
- Phase 2：固定结构，在 CDL 信道下微调 20 epochs
- 原因：CDL 信道不可微分，无法端到端训练，需先学好基础表征再适应真实信道

**AWGN 信道训练结果（SNR sweep）：**
```
SNR(dB)   ld=128   ld=256   ld=512
  -5      10.99    12.79    14.48
   0      14.84    16.76    18.55
   5      18.40    19.98    21.61
  10      20.56    21.71    23.13
  15      21.47    22.41    23.73
  20      21.79    22.66    23.93
  25      21.90    22.74    23.99
无信道上界  21.79    23.74    23.98
```

**CDL-A 信道微调结果（SNR sweep）：**
```
SNR(dB)   ld=128   ld=256   ld=512
  -5      11.73    12.39    13.34
   0      14.12    15.14    16.69
   5      16.45    17.86    19.45
  10      17.09    18.61    20.30
  15      17.33    18.87    20.50
  20      17.40    18.91    20.57
  25      17.40    18.94    20.59
```

**传统 baseline（JPEG quality=75 + LDPC5G + AWGN）：**
```
SNR(dB)   PSNR      成功率
  -5       0.00     0/500   ← 完全失败
   0       0.00     0/500   ← 完全失败
   5       9.92     1/500   ← 几乎失败
  10      30.52   500/500   ← 突然跳到高质量（悬崖效应）
  15+     30.52   500/500
```

**关键发现：**
1. 语义系统在 SNR < 10dB 时明显优于传统方法（克服悬崖效应）
2. latent dimension 越大，性能越好，低 SNR 提升更显著
3. CDL 比 AWGN 低约 3-4 dB，但差距随 latent dimension 增大而缩小
4. 两阶段训练比 CDL 从头训练提升约 3-4 dB

**权重文件路径：**
```
AWGN ld=128: checkpoints/image-jscc/recon_ld128_2026-03-25_02-19-35/best_psnr21.79.weights.h5
AWGN ld=256: checkpoints/image-jscc/recon_ld256_2026-03-25_01-02-27/best_psnr22.66.weights.h5
AWGN ld=512: checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5
CDL  ld=128: checkpoints/image-jscc/recon_cdl_finetune_2026-03-24_08-35-49/best_psnr17.09.weights.h5
CDL  ld=256: checkpoints/image-jscc/recon_cdl_finetune_ld256_2026-03-25_01-47-36/best_psnr18.59.weights.h5
CDL  ld=512: checkpoints/image-jscc/recon_cdl_finetune_ld512_2026-03-24_22-34-20/best_psnr20.20.weights.h5
```

**评估结果 JSON 文件：**
```
checkpoints/image-jscc/eval/snr_recon_ld128.json
checkpoints/image-jscc/eval/snr_recon_ld256.json
checkpoints/image-jscc/eval/snr_recon_ld512.json
checkpoints/image-jscc/eval/snr_recon_cdl.json
checkpoints/image-jscc/eval/snr_recon_cdl_ld256.json
checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json
checkpoints/image-jscc/eval/jpeg_baseline.json
```

**最终对比图：**
```
checkpoints/image-jscc/eval/final_v5.png
（六条语义曲线 + JPEG+LDPC，PSNR 和 SSIM 两张子图）
```

---

## 5. 项目完成标准（5项）
```
✅ 已完成
1. 原项目复现：BART + CDL-A，BLEU=0.935
2. 图像语义重建系统：AWGN + CDL 下稳定训练和评估
3. 消融实验：latent dimension（128/256/512）× 信道（AWGN/CDL）
4. 传统 baseline：JPEG+LDPC，悬崖效应验证
5. 对比图表：六条曲线，核心实验图

🔜 待完成
6. Sionna RT site-specific 实验（最高优先级，在 Google Colab 上做）
7. Kodak 数据集实验（可选，提升学术价值）
8. ViT/Transformer encoder 对比（可选，配合 Kodak 一起做）
```

---

## 6. 下一步：Sionna RT 实验

**为什么要在 Colab 上做：**
Sionna RT 模块从 0.14 版本才有，本地是 0.11.0 不含 RT。CDL 信道在 WSL2 下需要 `TF_FORCE_GPU_ALLOW_GROWTH=true`，RT 同样有 CUDA 依赖，Colab 环境更干净。

**实验设计思路（思路B，直接评估）：**
不需要重新训练，用已有的 CDL 微调权重（ld=512）直接在 RT 场景下评估。展示不同用户位置下的 PSNR 分布，这就是"site-specific"的核心含义。

**具体步骤：**
1. 打开 Google Colab，选择 T4 GPU 运行时
2. 安装新版 Sionna：`!pip install sionna`
3. 上传 `colab_rt.zip`（位于 Windows Documents 文件夹，44MB）
4. 解压，加载 ld=512 AWGN 模型权重
5. 加载 Sionna RT 内置场景（Munich 或 simple_street_canyon）
6. 在场景中设置基站和多个用户位置（10-20个）
7. 对每个位置用 RT 计算信道，再评估 PSNR
8. 画出：场景俯视图 + 用户位置 vs PSNR 曲线

**Colab 上传文件：**
```
Windows Documents/colab_rt.zip 包含：
- train_reconstruction.py
- checkpoints/image-jscc/recon_ld512_2026-03-24_21-40-03/best_psnr23.32.weights.h5
- checkpoints/image-jscc/eval/snr_recon_ld512.json
```

---

## 7. 后续计划（RT 完成后）

**Kodak 实验设计：**
- 训练集：DIV2K（800张）+ Flickr2K（2650张）= DF2K
- 测试集：Kodak（24张，标准 benchmark）
- 模型：需要重新设计 encoder/decoder 支持 256×256 patch 输入
- 同时训练 CNN 和 ViT（SegFormer-B0）两个版本，形成编码器消融实验
- 对比传统方法：JPEG+LDPC 和 BPG+LDPC

**论文实验结构：**
1. 系统验证（CIFAR-10，已完成）：证明系统有效，消融分析
2. 主要性能评估（Kodak，待做）：与文献对比的核心结果
3. Site-specific 验证（Sionna RT，待做）：项目独特贡献

---

## 8. 重要技术说明

**两阶段训练为什么必要：**
CDL 信道含有 LDPC 硬判决，不可微分，梯度无法从 decoder 传回 encoder。直接用 CDL 从头训练，encoder 接收不到梯度，性能很差（实验验证：13.45 dB vs 两阶段的 17+ dB）。先用可微分的 AnalogAWGN 预训练，让 encoder 学会基础语义压缩能力，再在 CDL 下微调，解决了这个问题。

**没有使用 VQ 的原因：**
原项目的 VQ 是为文字任务设计的，接收端有 BART decoder 的语言模型先验可以从损坏的特征中恢复语义。图像没有这种先验，VQ 量化索引经信道损坏后图像完全无法重建（实验验证：所有 SNR 下 Acc≈10%）。改用连续特征的 Analog 传输解决了这个问题，这是一个重要的架构决策。

**latent dimension 的理解：**
latent_dim=128 对应压缩比 24:1（3072→128），latent_dim=512 对应压缩比 6:1。更大的 latent 不只是性能更好，还让系统对信道噪声更鲁棒（每个维度携带的信息量更少，单个维度损坏对重建的影响更小）。

---

## 9. 消融实验结论（可直接写入论文）

1. **latent dimension 消融：** ld=128→256→512 在所有 SNR 下 PSNR 单调递增，低 SNR 区域（0dB）提升最大（+3.71 dB），说明压缩率是影响语义通信质量的核心超参数。

2. **信道类型消融：** CDL-A 比 AWGN 低约 3-4 dB，但差距随 latent_dim 增大而缩小（ld=128 差 4.39 dB，ld=512 差 3.40 dB），说明更大的 latent dimension 带来更强的信道鲁棒性。

3. **训练策略消融：** AWGN 预训练→CDL 微调比直接 CDL 训练提升约 3-4 dB（13.45 dB vs 17.09 dB），验证了两阶段训练策略的必要性。

4. **语义 vs 传统对比：** SNR < 10 dB 时，语义系统（所有配置）均优于 JPEG+LDPC，在 SNR = 0 dB 时语义系统 PSNR 约 15-18 dB，而 JPEG+LDPC 完全失败（PSNR=0）。

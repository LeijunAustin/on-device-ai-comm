# Image Semantic Communication Project — 项目状态文档
> 最后更新：2026年4月21日
> 用途：在新对话中恢复项目上下文，让 Claude 能继续帮助推进项目

---

## 0. 如何使用这份文档

把这份文档的内容粘贴到新对话的开头，然后告诉 Claude："这是我的项目状态文档，请帮我继续推进。"

---

## 1. 项目背景

**题目：** Site-specific Semantic E2E PUSCH Encoding/Decoding for Image Transmission

**参考仓库：** https://github.com/abman23/on-device-ai-comm（已 fork 到 https://github.com/LeijunAustin/on-device-ai-comm）

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

**每次开始工作前激活环境：**

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

整理后的项目文件分为三类：原项目文件、你新建的图像通信文件、以及 Colab RT 实验文件。

**原项目文件（语言语义通信，未改动或小改）：**
- `train.py`：原项目 BART 语言通信训练入口（Hugging Face Transformers）
- `eval.py`：原项目 BART 评估入口（计算 BLEU/ROUGE 分数）
- `models/on_device_ai_comm.py`：BART 语言模型定义
- `models/vq_vae.py`：向量量化器
- `models/channels.py`：信道模块，有两处修改：加了 `ebno_db_override=None` 参数，去掉了 XLA 编译
- `models/utils.py`：工具函数
- `scripts/train.sh`、`scripts/eval.sh`：原项目训练/评估脚本
- `train/args.py`：原项目命令行参数定义
- `preprocess/`：原项目数据预处理

**你新建的图像语义通信文件（项目核心贡献）：**
- `models/image_semantic_comm.py`：图像语义模型定义（CNN encoder/decoder）
- `train_reconstruction.py`：**核心训练脚本**，定义了模型结构、损失函数、两阶段训练逻辑，其他脚本都 import 这个文件
- `train_cdl_finetune.py`：CDL-A 信道微调脚本（两阶段训练第二阶段）
- `eval_reconstruction.py`：AWGN 信道 SNR sweep 评估
- `eval_cdl_snr.py`：CDL 信道 SNR sweep 评估
- `eval_jpeg_baseline.py`：JPEG+LDPC 传统方法基准评估
- `plot_results_final.py`：生成最终论文对比图（六条曲线）
- `run_overnight.sh`：批量运行所有实验的脚本（参考用）
- `CLAUDE.md`：本文档

**本地 RT 实验文件（2026-04-21 迁移完成）：**
- `run_rt_local.py`：Sionna RT site-specific 实验脚本，在本地 `sionna-new` 环境运行
- `RT.ipynb`：Colab 版本（备用），末尾 Cell 11 可将结果保存到 Drive

**可视化脚本：**
- `visualize_reconstruction.py`：展示不同 SNR 下的重建图像对比网格

---

## 4. 完整运行指南

### 前置：每次开始工作前必须激活环境

```bash
conda activate on-device-ai-comm
cd ~/on-device-ai-comm
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export DRJIT_LIBLLVM_PATH=/usr/lib/llvm-14/lib/libLLVM-14.so
```

---

### 工作流概览

整个实验是一条单向流水线，必须按顺序执行。每个步骤的输出路径就是下一个步骤的输入路径：

```
[Step 1] train_reconstruction.py
         读取：CIFAR-10（自动下载到 ~/.keras/datasets/）
         写入：checkpoints/image-jscc/recon_ld{N}_{时间戳}/best_psnr{xx.xx}.weights.h5
                                    ↓
[Step 2] train_cdl_finetune.py
         读取：checkpoints/image-jscc/recon_ld{N}_*/best_psnr*.weights.h5（自动查找）
         写入：checkpoints/image-jscc/recon_cdl_finetune_ld{N}_{时间戳}/best_psnr{xx.xx}.weights.h5
                                    ↓
[Step 3a] eval_reconstruction.py    读取 Step 1 权重 → 写入 eval/snr_recon_ld{N}.json
[Step 3b] eval_cdl_snr.py          读取 Step 2 权重 → 写入 eval/snr_recon_cdl[_ld{N}].json
[Step 3c] eval_jpeg_baseline.py    无需权重          → 写入 eval/jpeg_baseline.json
                                    ↓
[Step 4]  plot_results_final.py
          读取：checkpoints/image-jscc/eval/ 下的全部 7 个 JSON 文件
          写入：checkpoints/image-jscc/eval/final_v5.png
```

---

### Step 1：AWGN 预训练

**脚本：** `~/on-device-ai-comm/train_reconstruction.py`
**作用：** 用可微分的 AWGN 信道从头训练 CNN encoder + decoder，让模型学会基础的图像语义压缩能力。这是整个项目最重要的一步，后续所有实验都依赖这里训练出的权重。CDL 信道不可微分，所以必须先做这一步，才能进行 Step 2 的 CDL 微调。
**调用关系：** `train_reconstruction.py` 内部 import 了 `models/image_semantic_comm.py`（模型结构）和 `models/channels.py`（信道定义），不依赖任何其他脚本。

```bash
# ── 正式训练：三种 latent dim 各跑一次，约 2-4 小时/次 ─────────────

python train_reconstruction.py --latent-dim 128 --epochs 100
# 输出：checkpoints/image-jscc/recon_ld128_{时间戳}/best_psnr21.79.weights.h5
#       checkpoints/image-jscc/recon_ld128_{时间戳}/history.json

python train_reconstruction.py --latent-dim 256 --epochs 100
# 输出：checkpoints/image-jscc/recon_ld256_{时间戳}/best_psnr22.66.weights.h5
#       checkpoints/image-jscc/recon_ld256_{时间戳}/history.json

python train_reconstruction.py --latent-dim 512 --epochs 100
# 输出：checkpoints/image-jscc/recon_ld512_{时间戳}/best_psnr23.92.weights.h5
#       checkpoints/image-jscc/recon_ld512_{时间戳}/history.json

# ── 可选参数（均有默认值，不填也能跑）────────────────────────────────
#   --batch-size 64        每批图片数（默认 64）
#   --lr 5e-4              学习率（默认 5e-4）
#   --ebno-db-min 0        训练时 SNR 随机采样下限，单位 dB（默认 0）
#   --ebno-db-max 20       训练时 SNR 随机采样上限，单位 dB（默认 20）
#   --output-dir checkpoints/image-jscc   权重保存的父目录

# ── 冒烟测试：1 epoch 验证流程无报错，约 1-2 分钟 ────────────────────
python train_reconstruction.py --latent-dim 128 --epochs 1
# 期望输出末尾：Best PSNR: xx.xx dB → checkpoints/image-jscc/recon_ld128_.../
```

---

### Step 2：CDL 微调

**脚本：** `~/on-device-ai-comm/train_cdl_finetune.py`
**作用：** 在 CDL-A 真实衰落信道（含 LDPC 编解码）下对 Step 1 的模型进行微调。CDL 信道含有 LDPC 硬判决，不可微分，所以不能从头训练，只能从 AWGN 预训练权重出发微调。跳过 Step 1 直接跑 Step 2 会导致 encoder 收不到梯度，性能极差（实测 13.45 dB vs 两阶段的 17+ dB）。
**调用关系：** import 了 `train_reconstruction.py` 中的 `ReconstructionModel`、`load_cifar10`、`reconstruction_loss`、`psnr`、`ssim_metric`。权重路径通过 `glob` 自动查找 `checkpoints/image-jscc/recon_ld{N}_*/best_psnr*.weights.h5`，取 PSNR 数值最高的文件，不需要手动填写路径。

```bash
# ── 正式微调：三种 latent dim 各跑一次，约 30-60 分钟/次 ─────────────

python train_cdl_finetune.py --latent-dim 128
# 自动读取：checkpoints/image-jscc/recon_ld128_*/best_psnr*.weights.h5（最高 PSNR）
# 输出：    checkpoints/image-jscc/recon_cdl_finetune_ld128_{时间戳}/best_psnr17.09.weights.h5
#           checkpoints/image-jscc/recon_cdl_finetune_ld128_{时间戳}/history.json

python train_cdl_finetune.py --latent-dim 256
# 自动读取：checkpoints/image-jscc/recon_ld256_*/best_psnr*.weights.h5
# 输出：    checkpoints/image-jscc/recon_cdl_finetune_ld256_{时间戳}/best_psnr18.59.weights.h5

python train_cdl_finetune.py --latent-dim 512
# 自动读取：checkpoints/image-jscc/recon_ld512_*/best_psnr*.weights.h5
# 输出：    checkpoints/image-jscc/recon_cdl_finetune_ld512_{时间戳}/best_psnr20.20.weights.h5

# ── 可选参数 ─────────────────────────────────────────────────────────
#   --epochs 20    微调轮数（默认 20，CDL 收敛快，20 轮通常已足够）
#   --lr 1e-4      学习率（默认 1e-4，比 AWGN 预训练的 5e-4 小，避免破坏特征）

# ── 冒烟测试：约 2-3 分钟 ────────────────────────────────────────────
python train_cdl_finetune.py --latent-dim 512 --epochs 1
# 期望输出末尾：Done. Best CDL PSNR (ld=512): xx.xx dB → checkpoints/.../
```

---

### Step 3a：AWGN 信道评估

**脚本：** `~/on-device-ai-comm/eval_reconstruction.py`
**作用：** 在 SNR = [-5, 0, 5, 10, 15, 20, 25] dB 共 7 个点上逐一评估 PSNR 和 SSIM，同时评估无信道上界（bypass 模式，理论性能天花板）。结果保存为 JSON 供 Step 4 绘图使用。
**调用关系：** import 了 `train_reconstruction.py` 中的 `ReconstructionModel`。`--weights` 路径需要手动指定，填入 Step 1 生成的实际文件名。

```bash
# ── ld=128 ────────────────────────────────────────────────────────────
python eval_reconstruction.py \
    --weights checkpoints/image-jscc/recon_ld128_2026-03-25_02-19-35/best_psnr21.79.weights.h5 \
    --latent-dim 128
# 输出：checkpoints/image-jscc/eval/snr_recon_ld128.json

# ── ld=256 ────────────────────────────────────────────────────────────
python eval_reconstruction.py \
    --weights checkpoints/image-jscc/recon_ld256_2026-03-25_01-02-27/best_psnr22.66.weights.h5 \
    --latent-dim 256
# 输出：checkpoints/image-jscc/eval/snr_recon_ld256.json

# ── ld=512 ────────────────────────────────────────────────────────────
python eval_reconstruction.py \
    --weights checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5 \
    --latent-dim 512
# 输出：checkpoints/image-jscc/eval/snr_recon_ld512.json

# ── 可选参数 ─────────────────────────────────────────────────────────
#   --snr-min -5            SNR 起点，单位 dB（默认 -5）
#   --snr-max 25            SNR 终点，单位 dB（默认 25）
#   --snr-step 5            SNR 步长，单位 dB（默认 5）
#   --batch-size 64
#   --output-dir checkpoints/image-jscc/eval
```

---

### Step 3b：CDL 信道评估

**脚本：** `~/on-device-ai-comm/eval_cdl_snr.py`
**作用：** 在相同的 7 个 SNR 点上评估 CDL 微调权重的 PSNR 和 SSIM，用于和 Step 3a 的 AWGN 结果对比，量化真实衰落信道带来的性能损失（约 3-4 dB）。
**调用关系：** import 了 `train_reconstruction.py` 中的 `ReconstructionModel`、`psnr`、`ssim_metric`。通过 `glob` 自动查找 AWGN 权重（`recon_ld{N}_*/best_psnr*.weights.h5`）和 CDL 权重（`recon_cdl_finetune_ld{N}_*/best_psnr*.weights.h5`），均取 PSNR 数值最高的文件，不需要手动填路径。

```bash
python eval_cdl_snr.py --latent-dim 128
# 自动读取：checkpoints/image-jscc/recon_ld128_*/best_psnr*.weights.h5
#           checkpoints/image-jscc/recon_cdl_finetune_ld128_*/best_psnr*.weights.h5
# 输出：    checkpoints/image-jscc/eval/snr_recon_cdl.json  ← ld=128 用此历史命名

python eval_cdl_snr.py --latent-dim 256
# 输出：checkpoints/image-jscc/eval/snr_recon_cdl_ld256.json

python eval_cdl_snr.py --latent-dim 512
# 输出：checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json
```

---

### Step 3c：JPEG+LDPC 基准评估

**脚本：** `~/on-device-ai-comm/eval_jpeg_baseline.py`
**作用：** 模拟传统图像传输流水线（JPEG 压缩 → LDPC 编码 → AWGN 信道 → LDPC 解码 → JPEG 解压），在相同 SNR 条件下评估 PSNR，展示传统方法的"悬崖效应"：SNR < 10 dB 时几乎完全失败（PSNR = 0），SNR ≥ 10 dB 时突然跳到高质量（PSNR ≈ 30 dB）。
**调用关系：** 不依赖 `train_reconstruction.py`。用到了 Sionna 0.x 的 `sionna.channel.AWGN`、`sionna.fec.ldpc`、`sionna.mapping`（本地 Sionna 0.11.0 支持，无需 Colab）。

```bash
python eval_jpeg_baseline.py
# 无需任何参数，内部固定使用 CIFAR-10 测试集前 500 张，JPEG quality=75，LDPC k=512 n=1024
# 输出：checkpoints/image-jscc/eval/jpeg_baseline.json
# 耗时：约 10-20 分钟
```

---

### Step 4：生成论文对比图

**脚本：** `~/on-device-ai-comm/plot_results_final.py`
**作用：** 读取 Step 3 生成的全部 7 个 JSON 文件，绘制六条语义通信曲线（ld=128/256/512 × AWGN/CDL）和一条 JPEG+LDPC 基线，输出 PSNR 和 SSIM 两个子图的论文图。
**调用关系：** 不依赖任何模型代码，只读以下 7 个 JSON 文件：

```
checkpoints/image-jscc/eval/snr_recon_ld128.json
checkpoints/image-jscc/eval/snr_recon_ld256.json
checkpoints/image-jscc/eval/snr_recon_ld512.json
checkpoints/image-jscc/eval/snr_recon_cdl.json
checkpoints/image-jscc/eval/snr_recon_cdl_ld256.json
checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json
checkpoints/image-jscc/eval/jpeg_baseline.json
```

```bash
python plot_results_final.py
# 无需任何参数，路径均硬编码在脚本内
# 输出：checkpoints/image-jscc/eval/final_v5.png
# 耗时：几秒# 几秒即完成
```

---

## 5. 已完成实验结果

### 5.1 原项目复现（文字语义通信）

- 信道：CDL-A，SNR 5-15 dB
- BLEU = 0.935，rouge1 = 99.74%，BER = 0.128 @4dB
- 权重：`checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15/tf_model.h5`

### 5.2 图像重建系统（主要贡献）

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

**最佳权重文件路径：**

```
AWGN ld=128: checkpoints/image-jscc/recon_ld128_2026-03-25_02-19-35/best_psnr21.79.weights.h5
AWGN ld=256: checkpoints/image-jscc/recon_ld256_2026-03-25_01-02-27/best_psnr22.66.weights.h5
AWGN ld=512: checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5
CDL  ld=128: checkpoints/image-jscc/recon_cdl_finetune_ld128_2026-03-24_08-35-49/best_psnr17.09.weights.h5
CDL  ld=256: checkpoints/image-jscc/recon_cdl_finetune_ld256_2026-03-25_01-47-36/best_psnr18.59.weights.h5
CDL  ld=512: checkpoints/image-jscc/recon_cdl_finetune_ld512_2026-03-24_22-34-20/best_psnr20.20.weights.h5
```

**评估结果 JSON 文件（checkpoints/image-jscc/eval/）：**

```
snr_recon_ld128.json          AWGN ld=128
snr_recon_ld256.json          AWGN ld=256
snr_recon_ld512.json          AWGN ld=512
snr_recon_cdl.json            CDL  ld=128（历史兼容命名）
snr_recon_cdl_ld256.json      CDL  ld=256
snr_recon_cdl_ld512.json      CDL  ld=512
jpeg_baseline.json            JPEG+LDPC baseline
```

**最终对比图：** `checkpoints/image-jscc/eval/final_v5.png`

**重建图像可视化：** `checkpoints/image-jscc/eval/reconstruction_visual.png`

**RT 结果图（2000 采样点）：**
- `checkpoints/image-jscc/eval/site_specific_results_munich.png`
- `checkpoints/image-jscc/eval/site_specific_heatmap_munich.png`
- `checkpoints/image-jscc/eval/site_specific_results_sydney.png`
- `checkpoints/image-jscc/eval/site_specific_heatmap_sydney.png`

### 5.3 Sionna RT Site-Specific 实验（✅ 2026年3月31日完成）

```
场景：Munich（Sionna RT 内置，真实慕尼黑建筑几何），3.5 GHz
有效 UE 位置：433 个，距基站 26~402 m
SNR 范围：   -10.0 ~ 24.9 dB
PSNR 范围：   10.49 ~ 23.83 dB（跨度 13.3 dB）

**Munich（2026-04-21 更新，2000 采样目标，1744 有效位置）：**
近距离 (<100m)，n=503：   平均 PSNR = 20.7 dB
中距离 (100-250m)，n=1006：平均 PSNR = 18.9 dB
远距离 (>250m)，n=235：    平均 PSNR = 16.7 dB
★ 近距 vs 远距 PSNR 差 = 4.0 dB

**Sydney CBD（2026-04-21 新增，2000 采样目标，1982 有效位置）：**
近距离 (<100m)，n=118：   平均 PSNR = 20.4 dB
中距离 (100-250m)，n=1489：平均 PSNR = 14.6 dB
远距离 (>250m)，n=375：    平均 PSNR = 11.9 dB
★ 近距 vs 远距 PSNR 差 = 8.5 dB

**University of Sydney（2026-04-21 新增，2000 采样目标，1998 有效位置）：**
场景范围：819m × 682m，建筑最高 37.5m，基站位于 [16.0, 36.4, 30.0]
近距离 (<100m)，n=233：   平均 PSNR = 18.0 dB
中距离 (100-250m)，n=938：平均 PSNR = 11.7 dB
远距离 (>250m)，n=827：   平均 PSNR = 10.8 dB
★ 近距 vs 远距 PSNR 差 = 7.2 dB
场景文件：`/mnt/e/OneDrive/Desktop/usyd_scene/usyd_campus.xml`（由 convert_to_mitsuba.py 生成）
```

---

## 6. 项目完成标准

```
✅ 已完成
1. 原项目复现：BART + CDL-A，BLEU=0.935
2. 图像语义重建系统：AWGN + CDL 下稳定训练和评估
3. 消融实验：latent dimension（128/256/512）× 信道（AWGN/CDL）
4. 传统 baseline：JPEG+LDPC，悬崖效应验证
5. 对比图表：六条曲线，核心实验图
6. Sionna RT site-specific 实验：Munich 场景，433 个 UE 位置，近远距 PSNR 差 4.8 dB

🔜 可选（提升学术价值）
7. Kodak 数据集实验
8. ViT/Transformer encoder 对比
```

---

## 7. Sionna RT 实验（Google Colab）

### 本地运行（推荐，2026-04-21 迁移完成）

RT 实验现在可以在本地 `sionna-new` 环境运行，不再依赖 Colab：

```bash
# 内置场景（无需额外文件）
conda run -n sionna-new python run_rt_local.py --scene munich
conda run -n sionna-new python run_rt_local.py --scene etoile

# 自定义场景（首次需要 zip，之后自动使用缓存）
conda run -n sionna-new python run_rt_local.py --scene sydney \
    --sydney-zip /mnt/e/OneDrive/Desktop/sydney_scene.zip

# 强制重新计算（忽略缓存）
conda run -n sionna-new python run_rt_local.py --scene munich --force-recompute
```

CIR 缓存路径：`checkpoints/image-jscc/rt_cache/{scene}_{a,tau,ue_pos}.npy`

**两个环境说明：**
- `on-device-ai-comm`（Python 3.9, TF 2.10, Sionna 0.11）：训练和评估
- `sionna-new`（Python 3.10, TF 2.21, Sionna 1.2.2）：RT 实验专用

两个环境通过文件交换数据（权重 .h5 → sionna-new 读取；JSON → on-device-ai-comm 绘图），互不干扰。

### Colab 快速重连指南

每次重新打开 Colab session，按顺序运行以下步骤。

**Cell 1：安装 Sionna**

```python
import sys, os
try:
    import sionna.rt
    print(f"Sionna 已就绪: {sionna.__version__}")
except ImportError:
    os.system("pip install -q sionna")
    os.kill(os.getpid(), 5)  # 触发 Runtime 重启，重启后从 Cell 2 继续
```

**Cell 2：从 Google Drive 挂载并解压（建议把 colab_rt.zip 上传到 Drive）**

```python
import os, sys, zipfile, glob
from google.colab import drive

drive.mount('/content/drive')
ZIP_PATH = "/content/drive/MyDrive/jscc_project/colab_rt.zip"
WORK_DIR = "/content/jscc"

if not os.path.exists(WORK_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(WORK_DIR)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# 必须修复：清空 __init__.py，防止触发 BART 模型的 import 链
with open(f"{WORK_DIR}/models/__init__.py", "w") as f:
    f.write("# intentionally empty\n")

# 必须修复：替换 channels.py 为 Sionna 1.x 兼容版
# （channels_stub 代码见 colab_rt_v3.py Cell 2）

weight_files = glob.glob("checkpoints/image-jscc/recon_ld512_*/best_psnr*.weights.h5")
WEIGHTS_PATH = sorted(weight_files)[-1]
print(f"权重: {WEIGHTS_PATH}")
```

**Cell 3-10：** 按顺序运行 `colab_rt_v3.py` 的各个 Cell。

### Colab 环境踩坑记录

**`ModuleNotFoundError: No module named 'models.channels'`** → 清空 `models/__init__.py`（Cell 2 已修复）。

**`ModuleNotFoundError: No module named 'sionna.channel'`** → 用 Sionna 1.x 兼容 stub 覆盖 `channels.py`（Cell 2 已修复）。

**`ValueError: A total of 33 objects could not be loaded`** → Keras 2（本地）→ Keras 3（Colab）格式不兼容，使用 `load_weights_correct()` 函数按"层类型 + 数字编号 + weight_type"三重键分组排序后按位置对应赋值（见 `colab_rt_v3.py` Cell 4）。

**`ValueError: operands could not be broadcast (7,2) and (6,2)`** → Sionna 1.x 的 `paths.cir()` 返回 6 维而非 7 维，pad 操作改用动态维度（见 `colab_rt_v3.py` Cell 7）。

**Sionna 1.x 关键 API 差异：**
- `PathSolver()(scene, ...)` 替代 `scene.compute_paths()`
- `RadioMapSolver()(scene, ...)` 替代 `scene.coverage_map()`
- `paths.cir(out_type='numpy')` 返回框架无关的 numpy 数组
- `a` 实际形状：`(batch, num_rx=4, num_tx=1, num_rx_ant=16, num_paths, time)`（6 维）
- 上行转置：`a = np.transpose(a, (0,2,3,1,4,5))`，`tau = np.transpose(tau, (0,2,1))`

### colab_rt.zip 打包命令

```bash
cd ~/on-device-ai-comm
python - << 'EOF'
import zipfile, os, glob
files = [
    "train_reconstruction.py",
    *glob.glob("models/*.py"),
    "checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5",
    "checkpoints/image-jscc/eval/snr_recon_ld512.json",
    "checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json",
    "checkpoints/image-jscc/eval/jpeg_baseline.json",
]
with zipfile.ZipFile("colab_rt.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        if os.path.exists(f):
            zf.write(f)
            print(f"  添加: {f}")
print(f"完成: {os.path.getsize('colab_rt.zip')/1024/1024:.1f} MB")
EOF
```

---

## 8. 重要技术说明

**两阶段训练为什么必要：** CDL 信道含有 LDPC 硬判决，不可微分，梯度无法从 decoder 传回 encoder。直接用 CDL 从头训练，encoder 接收不到梯度，性能很差（实验验证：13.45 dB vs 两阶段的 17+ dB）。

**没有使用 VQ 的原因：** 原项目的 VQ 是为文字任务设计的，图像没有语言模型先验，VQ 量化索引经信道损坏后图像完全无法重建（所有 SNR 下 Acc≈10%）。改用连续特征的 Analog 传输解决了这个问题。

**latent dimension 的理解：** `latent_dim=128` 对应压缩比 24:1，`latent_dim=512` 对应压缩比 6:1。更大的 latent 不只是性能更好，还让系统对信道噪声更鲁棒。

---

## 9. 消融实验结论（可直接写入论文）

1. **latent dimension 消融：** ld=128→256→512 在所有 SNR 下 PSNR 单调递增，低 SNR 区域（0dB）提升最大（+3.71 dB），说明压缩率是影响语义通信质量的核心超参数。

2. **信道类型消融：** CDL-A 比 AWGN 低约 3-4 dB，但差距随 latent_dim 增大而缩小（ld=128 差 4.39 dB，ld=512 差 3.40 dB），说明更大的 latent dimension 带来更强的信道鲁棒性。

3. **训练策略消融：** AWGN 预训练→CDL 微调比直接 CDL 训练提升约 3-4 dB（13.45 dB vs 17.09 dB），验证了两阶段训练策略的必要性。

4. **语义 vs 传统对比：** SNR < 10 dB 时，语义系统均优于 JPEG+LDPC。SNR = 0 dB 时语义系统 PSNR 约 15-18 dB，而 JPEG+LDPC 完全失败（PSNR=0）。

5. **Site-specific 空间差异：** Munich 场景中近距离用户平均 PSNR 比远距离用户高 4.8 dB，且同距离下 PSNR 方差超过 10 dB，体现了真实建筑遮挡造成的地理位置相关特性，是 CDL 等随机信道模型无法捕捉的。

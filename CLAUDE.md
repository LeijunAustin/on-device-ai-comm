# Image Semantic Communication Project — 项目状态文档
> 最后更新：2026年3月31日
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
- `models/image_semantic_comm.py`：图像语义模型定义（CNN encoder/decoder）
- `train_reconstruction.py`：图像重建训练脚本，支持 analog/CDL 两种信道，两阶段训练
- `train_cdl_finetune.py`：ld=128 CDL 微调脚本
- `train_cdl_finetune_ld512.py`：ld=512 CDL 微调脚本
- `eval_reconstruction.py`：AWGN 信道 SNR sweep 评估
- `eval_cdl_snr.py`：CDL 信道 SNR sweep 评估（ld=128）
- `eval_cdl_snr_ld512.py`：CDL 信道 SNR sweep 评估（ld=512）
- `eval_jpeg_baseline.py`：JPEG+LDPC 传统 baseline 评估
- `plot_final_v5.py`：生成最终六条曲线对比图
- `colab_rt_v3.py`：Sionna RT site-specific 实验脚本（在 Google Colab 上运行）

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

### 4.3 Sionna RT Site-Specific 实验（✅ 2026年3月31日完成）

**实验设计（思路B，直接评估）：**
不重新训练，直接用已有的 AWGN ld=512 权重，通过 RT 提取每个用户位置的等效接收 SNR，再通过预计算的 PSNR-SNR 查找表映射到语义 PSNR。

**实验配置：**
```
场景：    Sionna RT 内置 Munich 场景（慕尼黑市区，真实建筑几何）
频率：    3.5 GHz（5G Sub-6GHz 典型频段）
基站位置：(8.5, 21.0, 27.0) m，屋顶，高 27m
基站天线：16 天线，PlanarArray 1×8，tr38901 方向图，交叉极化
UE 天线： 4 天线，PlanarArray 1×2，全向，交叉极化
UE 采样： RadioMapSolver → rm.sample_positions()，距离 10-400m，路径增益 -130~0 dB
CIR 数量：500 个位置，最终有效 433 个（过滤全零 CIR）
射线追踪：max_depth=5，max_num_paths_per_src=10^6
```

**Sionna 1.x 关键 API 细节（与 0.x 不同）：**
```
- PathSolver()(scene, ...) 替代了 scene.compute_paths()
- RadioMapSolver()(scene, ...) 替代了 scene.coverage_map()
- paths.cir(out_type='numpy') 返回 numpy 数组，框架无关
- sionna.phy 基于 PyTorch，sionna.rt 基于 Dr.Jit（框架无关）
- a 的实际形状为 6 维：(batch, num_rx=4, num_tx=1, num_rx_ant=16, num_paths, time)
  而不是官方文档里的 7 维（num_tx_ant 维度被 squeeze）
- 上行转置：a = np.transpose(a, (0,2,3,1,4,5))，tau = np.transpose(tau, (0,2,1))
```

**Colab 环境问题及解决方案（重要，下次运行可直接参考）：**

Colab 安装的是 Sionna 1.x（TF 2.19 + Keras 3.13），而本地权重用 TF 2.10（Keras 2）训练。直接调用 `load_weights()` 会报"expected 2 variables, but received 0"的错误，原因是 Keras 2 和 Keras 3 的 h5 文件格式不兼容。

解决方案是自定义 `load_weights_correct()` 函数，用 `h5py` 直接读取权重文件，然后按"层类型 + 层编号数字 + weight_type"三重键分组排序后，与 Keras 3 模型变量按位置对应赋值。关键细节：h5 文件的 `visititems` 按字母顺序遍历（`conv2d_10` 排在 `conv2d_2` 前面），必须按数字大小重新排序，否则权重赋值顺序会错乱。正确加载后验证 PSNR 与本地实验吻合（误差 < 0.2 dB）。

此外还需要在解压 zip 后立即：1）清空 `models/__init__.py`（防止触发 BART 模型的 import 链）；2）用兼容 Sionna 1.x 的 stub 覆盖 `models/channels.py`（原版依赖已废弃的 `sionna.channel` 路径）；3）不导入 `sionna.phy`（只用 `sionna.rt`，避免与 TF 冲突）。

**实验结果：**
```
有效 UE 位置数：  433 个
SNR 范围：       -10.0 ~ 24.9 dB
PSNR 范围：       10.49 ~ 23.83 dB（跨度 13.3 dB）

按距离分组：
  近距离 (<100m)，n=125：   平均 PSNR = 20.4 dB
  中距离 (100-250m)，n=244：平均 PSNR = 19.1 dB
  远距离 (>250m)，n=64：    平均 PSNR = 15.5 dB

★ 近距 vs 远距 PSNR 差 = 4.8 dB
```

**输出文件：**
```
site_specific_results.json    — 433 个 UE 位置的详细结果
site_specific_results_v3.png  — 四子图论文图（俯视图/距离散点/SNR曲线/箱线图）
```

**colab_rt_v3.py 的打包内容（供下次重新运行参考）：**
```
colab_rt.zip 需包含：
  train_reconstruction.py
  models/（整个目录，包含 channels.py、image_semantic_comm.py、__init__.py 等）
  checkpoints/image-jscc/recon_ld512_2026-03-25_03-50-35/best_psnr23.92.weights.h5
  checkpoints/image-jscc/eval/snr_recon_ld512.json
  checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json
  checkpoints/image-jscc/eval/jpeg_baseline.json
```

---

## 5. 项目完成标准

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
8. ViT/Transformer encoder 对比（配合 Kodak 一起做）
```

---

## 6. 后续计划（可选）

**Kodak 实验设计：**
- 训练集：DIV2K（800张）+ Flickr2K（2650张）= DF2K
- 测试集：Kodak（24张，标准 benchmark）
- 模型：需要重新设计 encoder/decoder 支持 256×256 patch 输入
- 同时训练 CNN 和 ViT（SegFormer-B0）两个版本，形成编码器消融实验
- 对比传统方法：JPEG+LDPC 和 BPG+LDPC

**论文实验结构：**
1. 系统验证（CIFAR-10，已完成）：证明系统有效，消融分析
2. Site-specific 验证（Sionna RT，已完成）：项目独特贡献
3. 主要性能评估（Kodak，可选）：与文献对比的核心结果

---

## 7. 重要技术说明

**两阶段训练为什么必要：**
CDL 信道含有 LDPC 硬判决，不可微分，梯度无法从 decoder 传回 encoder。直接用 CDL 从头训练，encoder 接收不到梯度，性能很差（实验验证：13.45 dB vs 两阶段的 17+ dB）。先用可微分的 AnalogAWGN 预训练，让 encoder 学会基础语义压缩能力，再在 CDL 下微调，解决了这个问题。

**没有使用 VQ 的原因：**
原项目的 VQ 是为文字任务设计的，接收端有 BART decoder 的语言模型先验可以从损坏的特征中恢复语义。图像没有这种先验，VQ 量化索引经信道损坏后图像完全无法重建（实验验证：所有 SNR 下 Acc≈10%）。改用连续特征的 Analog 传输解决了这个问题，这是一个重要的架构决策。

**latent dimension 的理解：**
latent_dim=128 对应压缩比 24:1（3072→128），latent_dim=512 对应压缩比 6:1。更大的 latent 不只是性能更好，还让系统对信道噪声更鲁棒（每个维度携带的信息量更少，单个维度损坏对重建的影响更小）。

---

## 8. 消融实验结论（可直接写入论文）

1. **latent dimension 消融：** ld=128→256→512 在所有 SNR 下 PSNR 单调递增，低 SNR 区域（0dB）提升最大（+3.71 dB），说明压缩率是影响语义通信质量的核心超参数。

2. **信道类型消融：** CDL-A 比 AWGN 低约 3-4 dB，但差距随 latent_dim 增大而缩小（ld=128 差 4.39 dB，ld=512 差 3.40 dB），说明更大的 latent dimension 带来更强的信道鲁棒性。

3. **训练策略消融：** AWGN 预训练→CDL 微调比直接 CDL 训练提升约 3-4 dB（13.45 dB vs 17.09 dB），验证了两阶段训练策略的必要性。

4. **语义 vs 传统对比：** SNR < 10 dB 时，语义系统（所有配置）均优于 JPEG+LDPC，在 SNR = 0 dB 时语义系统 PSNR 约 15-18 dB，而 JPEG+LDPC 完全失败（PSNR=0）。

5. **Site-specific 空间差异：** 在 Munich 场景中，近距离用户（<100m）平均 PSNR 比远距离用户（>250m）高 4.8 dB，且同距离下 PSNR 方差超过 10 dB，说明建筑遮挡造成的 NLOS 效应对语义通信质量有显著影响，这是 CDL 等随机信道模型无法捕捉的 site-specific 特性。

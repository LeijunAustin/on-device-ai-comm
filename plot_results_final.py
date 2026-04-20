import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# SNR 对齐修正：JPEG+LDPC 使用 Eb/N0，DeepJSCC 使用 Es/N0
# Es/N0 = Eb/N0 + 10*log10(R * log2(M)) = Eb/N0 + 10*log10(0.5*4) = Eb/N0 + 3.01 dB
JPEG_SNR_OFFSET = 3.01   # dB，将 JPEG Eb/N0 轴右移对齐到 Es/N0

def load(path):
    with open(path) as f:
        return json.load(f)

ld128 = load('checkpoints/image-jscc/eval/snr_recon_ld128.json')
ld256 = load('checkpoints/image-jscc/eval/snr_recon_ld256.json')
ld512 = load('checkpoints/image-jscc/eval/snr_recon_ld512.json')
cdl128 = load('checkpoints/image-jscc/eval/snr_recon_cdl.json')
cdl256 = load('checkpoints/image-jscc/eval/snr_recon_cdl_ld256.json')
cdl512 = load('checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json')
jpeg = load('checkpoints/image-jscc/eval/jpeg_baseline.json')

# 加载固定 CBR JPEG 对比（若存在）
jpeg_fixed_cbr_path = 'checkpoints/image-jscc/eval/jpeg_fixed_cbr.json'
jpeg_fcbr = load(jpeg_fixed_cbr_path) if os.path.exists(jpeg_fixed_cbr_path) else None

kodak_path = 'checkpoints/image-jscc/eval/kodak_ld512.json'
kodak_ld512 = load(kodak_path) if os.path.exists(kodak_path) else None

def extract(data, key, snr_offset=0.0):
    snr  = [d['ebno_db'] + snr_offset for d in data]
    vals = [d[key] if d.get('success_rate', 1) > 0.5 else 0 for d in data]
    return snr, vals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# CBR 值
CBR = {128: 0.042, 256: 0.083, 512: 0.167}
JPEG_CBR = 1.24

colors = {
    'ld128_awgn':    ('b', 'o', '-'),
    'ld256_awgn':    ('c', 'D', '-'),
    'ld512_awgn':    ('m', 's', '-'),
    'ld128_cdl':     ('b', '^', '--'),
    'ld256_cdl':     ('c', 'v', '--'),
    'ld512_cdl':     ('m', 'P', '--'),
    'jpeg':          ('r', 's', '-'),
    'jpeg_fcbr':     ('orange', 'x', '-.'),
    'kodak_ld512':   ('darkviolet', 'h', '-'),
}

datasets = [
    (ld128,  'ld128_awgn', f'DeepJSCC ld=128, CBR={CBR[128]:.3f} (AWGN)',  0.0),
    (ld256,  'ld256_awgn', f'DeepJSCC ld=256, CBR={CBR[256]:.3f} (AWGN)',  0.0),
    (ld512,  'ld512_awgn', f'DeepJSCC ld=512, CBR={CBR[512]:.3f} (AWGN)',  0.0),
    (cdl128, 'ld128_cdl',  f'DeepJSCC ld=128, CBR={CBR[128]:.3f} (CDL-A)', 0.0),
    (cdl256, 'ld256_cdl',  f'DeepJSCC ld=256, CBR={CBR[256]:.3f} (CDL-A)', 0.0),
    (cdl512, 'ld512_cdl',  f'DeepJSCC ld=512, CBR={CBR[512]:.3f} (CDL-A)', 0.0),
    (jpeg,   'jpeg',       f'JPEG+LDPC, CBR≈{JPEG_CBR} (Eb/N0, +3dB corr.)', JPEG_SNR_OFFSET),
]
if jpeg_fcbr is not None:
    datasets.append((jpeg_fcbr, 'jpeg_fcbr',
                     f'JPEG+LDPC fixed CBR=0.167 (Eb/N0, +3dB corr.)', JPEG_SNR_OFFSET))
if kodak_ld512 is not None:
    datasets.append((kodak_ld512, 'kodak_ld512',
                     f'DeepJSCC ld=512, CBR={CBR[512]:.3f} (Kodak, 32×32 patches)', 0.0))

for data, key, label, offset in datasets:
    snr, psnr = extract(data, 'psnr', offset)
    _, ssim   = extract(data, 'ssim', offset)
    c, m, ls  = colors[key]
    lw = 2.5 if 'jpeg' in key else 2
    ax1.plot(snr, psnr, color=c, marker=m, ls=ls, lw=lw, ms=7, label=label)
    ax2.plot(snr, ssim, color=c, marker=m, ls=ls, lw=lw, ms=7, label=label)

# RT site-specific 散点
rt_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
rt_files = sorted(glob.glob('checkpoints/image-jscc/eval/site_specific_results_*.json'))
for i, rt_path in enumerate(rt_files):
    scene_name = os.path.basename(rt_path).replace('site_specific_results_', '').replace('.json', '')
    rt_data = load(rt_path)
    rt_snr  = [d['snr_db']  for d in rt_data]
    rt_psnr = [d['psnr_db'] for d in rt_data]
    ax1.scatter(rt_snr, rt_psnr, color=rt_colors[i % len(rt_colors)],
                s=8, alpha=0.25, zorder=2, label=f'RT site-specific ({scene_name})')

ax1.axhline(y=23.99, color='gray', ls=':', lw=1.5, label='No channel upper bound (ld=512)')
ax1.set_xlabel('SNR (Es/N0, dB) — JPEG+LDPC axis shifted +3.01 dB for alignment', fontsize=10)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR vs SNR — CIFAR-10', fontsize=13)
ax1.legend(fontsize=7.5, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 27)
ax1.set_ylim(0, 35)

ax2.set_xlabel('SNR (Es/N0, dB) — JPEG+LDPC axis shifted +3.01 dB for alignment', fontsize=10)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM vs SNR — CIFAR-10', fontsize=13)
ax2.legend(fontsize=7.5, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 27)
ax2.set_ylim(0, 1.05)

plt.suptitle('Image Semantic Communication vs Traditional (CIFAR-10)\n'
             'Ablation: Latent Dimension × Channel Type  |  '
             'JPEG+LDPC: QAM-16, LDPC rate=1/2, avg CBR=1.24', fontsize=11)
plt.tight_layout()
plt.savefig('checkpoints/image-jscc/eval/final_v6.png', dpi=150, bbox_inches='tight')
print("Saved: final_v6.png")

# 汇总表
print(f"\n{'SNR':>6} {'A128':>7} {'A256':>7} {'A512':>7} {'C128':>7} {'C256':>7} {'C512':>7} {'JPEG':>7}")
print("-" * 60)
snrs = [d['ebno_db'] for d in ld128]
for i, snr in enumerate(snrs):
    def g(data, i): return data[i]['psnr'] if data[i].get('success_rate',1)>0.5 else 0
    print(f"{snr:>6} {ld128[i]['psnr']:>7.2f} {ld256[i]['psnr']:>7.2f} {ld512[i]['psnr']:>7.2f} "
          f"{cdl128[i]['psnr']:>7.2f} {cdl256[i]['psnr']:>7.2f} {cdl512[i]['psnr']:>7.2f} "
          f"{g(jpeg,i):>7.2f}")

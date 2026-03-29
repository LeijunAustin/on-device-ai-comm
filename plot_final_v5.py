import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 加载所有数据
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

def extract(data, key):
    snr  = [d['ebno_db'] for d in data]
    vals = [d[key] if d.get('success_rate', 1) > 0.5 else 0 for d in data]
    return snr, vals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = {
    'ld128_awgn': ('b', 'o', '-'),
    'ld256_awgn': ('c', 'D', '-'),
    'ld512_awgn': ('m', 's', '-'),
    'ld128_cdl':  ('b', '^', '--'),
    'ld256_cdl':  ('c', 'v', '--'),
    'ld512_cdl':  ('m', 'P', '--'),
    'jpeg':       ('r', 's', '-'),
}

datasets = [
    (ld128,  'ld128_awgn', 'Semantic ld=128 (AWGN)'),
    (ld256,  'ld256_awgn', 'Semantic ld=256 (AWGN)'),
    (ld512,  'ld512_awgn', 'Semantic ld=512 (AWGN)'),
    (cdl128, 'ld128_cdl',  'Semantic ld=128 (CDL-A)'),
    (cdl256, 'ld256_cdl',  'Semantic ld=256 (CDL-A)'),
    (cdl512, 'ld512_cdl',  'Semantic ld=512 (CDL-A)'),
    (jpeg,   'jpeg',       'JPEG+LDPC (traditional)'),
]

for data, key, label in datasets:
    snr, psnr = extract(data, 'psnr')
    _, ssim   = extract(data, 'ssim')
    c, m, ls  = colors[key]
    lw = 2.5 if key == 'jpeg' else 2
    ax1.plot(snr, psnr, color=c, marker=m, ls=ls, lw=lw, ms=7, label=label)
    ax2.plot(snr, ssim, color=c, marker=m, ls=ls, lw=lw, ms=7, label=label)

ax1.axhline(y=23.99, color='gray', ls=':', lw=1.5, label='No channel upper bound (ld=512)')
ax1.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR vs SNR — CIFAR-10', fontsize=13)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 27)
ax1.set_ylim(0, 35)

ax2.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM vs SNR — CIFAR-10', fontsize=13)
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 27)
ax2.set_ylim(0, 1.05)

plt.suptitle('Image Semantic Communication vs Traditional (CIFAR-10)\n'
             'Ablation: Latent Dimension × Channel Type', fontsize=12)
plt.tight_layout()
plt.savefig('checkpoints/image-jscc/eval/final_v5.png', dpi=150, bbox_inches='tight')
print("Saved: final_v5.png")

# 汇总表
print(f"\n{'SNR':>6} {'A128':>7} {'A256':>7} {'A512':>7} {'C128':>7} {'C256':>7} {'C512':>7} {'JPEG':>7}")
print("-" * 60)
snrs = [d['ebno_db'] for d in ld128]
for i, snr in enumerate(snrs):
    def g(data, i): return data[i]['psnr'] if data[i].get('success_rate',1)>0.5 else 0
    print(f"{snr:>6} {ld128[i]['psnr']:>7.2f} {ld256[i]['psnr']:>7.2f} {ld512[i]['psnr']:>7.2f} "
          f"{cdl128[i]['psnr']:>7.2f} {cdl256[i]['psnr']:>7.2f} {cdl512[i]['psnr']:>7.2f} "
          f"{g(jpeg,i):>7.2f}")

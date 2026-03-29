import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 加载所有数据
with open('checkpoints/image-jscc/eval/snr_recon_ld128.json') as f:
    ld128_awgn = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_ld512.json') as f:
    ld512_awgn = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_cdl.json') as f:
    ld128_cdl = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_cdl_ld512.json') as f:
    ld512_cdl = json.load(f)
with open('checkpoints/image-jscc/eval/jpeg_baseline.json') as f:
    jpeg = json.load(f)

def extract(data, key):
    snr  = [d['ebno_db'] for d in data]
    vals = [d[key] if d.get('success_rate', 1) > 0.5 else 0 for d in data]
    return snr, vals

snr_a128, psnr_a128 = extract(ld128_awgn, 'psnr')
snr_a512, psnr_a512 = extract(ld512_awgn, 'psnr')
snr_c128, psnr_c128 = extract(ld128_cdl,  'psnr')
snr_c512, psnr_c512 = extract(ld512_cdl,  'psnr')
snr_jpg,  psnr_jpg  = extract(jpeg,        'psnr')

_, ssim_a128 = extract(ld128_awgn, 'ssim')
_, ssim_a512 = extract(ld512_awgn, 'ssim')
_, ssim_c128 = extract(ld128_cdl,  'ssim')
_, ssim_c512 = extract(ld512_cdl,  'ssim')
_, ssim_jpg  = extract(jpeg,        'ssim')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PSNR 图
ax1.plot(snr_a128, psnr_a128, 'b-o',  lw=2, ms=7, label='Semantic CNN ld=128 (AWGN)')
ax1.plot(snr_a512, psnr_a512, 'c-D',  lw=2, ms=7, label='Semantic CNN ld=512 (AWGN)')
ax1.plot(snr_c128, psnr_c128, 'g-^',  lw=2, ms=7, label='Semantic CNN ld=128 (CDL-A)')
ax1.plot(snr_c512, psnr_c512, 'm-v',  lw=2, ms=7, label='Semantic CNN ld=512 (CDL-A)')
ax1.plot(snr_jpg,  psnr_jpg,  'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax1.axhline(y=23.98, color='gray', ls=':', lw=1.5, label='No channel upper bound (ld=512, 23.98dB)')
ax1.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR vs SNR — CIFAR-10', fontsize=13)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 27)
ax1.set_ylim(0, 35)

# SSIM 图
ax2.plot(snr_a128, ssim_a128, 'b-o',  lw=2, ms=7, label='Semantic CNN ld=128 (AWGN)')
ax2.plot(snr_a512, ssim_a512, 'c-D',  lw=2, ms=7, label='Semantic CNN ld=512 (AWGN)')
ax2.plot(snr_c128, ssim_c128, 'g-^',  lw=2, ms=7, label='Semantic CNN ld=128 (CDL-A)')
ax2.plot(snr_c512, ssim_c512, 'm-v',  lw=2, ms=7, label='Semantic CNN ld=512 (CDL-A)')
ax2.plot(snr_jpg,  ssim_jpg,  'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax2.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM vs SNR — CIFAR-10', fontsize=13)
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 27)
ax2.set_ylim(0, 1.05)

plt.suptitle('Image Semantic Communication vs Traditional — CIFAR-10\n(CNN Encoder-Decoder, AWGN & CDL-A Channels)', fontsize=12)
plt.tight_layout()
plt.savefig('checkpoints/image-jscc/eval/final_v4.png', dpi=150, bbox_inches='tight')
print("Saved: final_v4.png")

# 汇总表
print(f"\n{'SNR':>6} {'A-128':>8} {'A-512':>8} {'C-128':>8} {'C-512':>8} {'JPEG':>8}")
print("-" * 55)
for i in range(len(snr_a128)):
    print(f"{snr_a128[i]:>6} {psnr_a128[i]:>8.2f} {psnr_a512[i]:>8.2f} "
          f"{psnr_c128[i]:>8.2f} {psnr_c512[i]:>8.2f} {psnr_jpg[i]:>8.2f}")

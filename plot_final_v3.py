import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

with open('checkpoints/image-jscc/eval/snr_recon_ld128.json') as f:
    ld128 = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_ld512.json') as f:
    ld512 = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_cdl.json') as f:
    cdl = json.load(f)
with open('checkpoints/image-jscc/eval/jpeg_baseline.json') as f:
    jpeg = json.load(f)

def extract(data, key, threshold=0.5):
    return [d['ebno_db'] for d in data], \
           [d[key] if d.get('success_rate', 1) > threshold else 0 for d in data]

snr_128, psnr_128 = extract(ld128, 'psnr')
snr_512, psnr_512 = extract(ld512, 'psnr')
snr_cdl, psnr_cdl = extract(cdl,   'psnr')
snr_jpg, psnr_jpg = extract(jpeg,   'psnr')

_, ssim_128 = extract(ld128, 'ssim')
_, ssim_512 = extract(ld512, 'ssim')
_, ssim_cdl = extract(cdl,   'ssim')
_, ssim_jpg = extract(jpeg,   'ssim')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(snr_128, psnr_128, 'b-o',  lw=2, ms=7, label='Semantic CNN ld=128 (AWGN)')
ax1.plot(snr_512, psnr_512, 'c-D',  lw=2, ms=7, label='Semantic CNN ld=512 (AWGN)')
ax1.plot(snr_cdl, psnr_cdl, 'g-^',  lw=2, ms=7, label='Semantic CNN ld=128 (CDL-A)')
ax1.plot(snr_jpg, psnr_jpg, 'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax1.axhline(y=23.98, color='gray', ls=':', lw=1.5, label='No channel upper bound (ld=512)')
ax1.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR vs SNR — CIFAR-10', fontsize=13)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 27)
ax1.set_ylim(0, 35)

ax2.plot(snr_128, ssim_128, 'b-o',  lw=2, ms=7, label='Semantic CNN ld=128 (AWGN)')
ax2.plot(snr_512, ssim_512, 'c-D',  lw=2, ms=7, label='Semantic CNN ld=512 (AWGN)')
ax2.plot(snr_cdl, ssim_cdl, 'g-^',  lw=2, ms=7, label='Semantic CNN ld=128 (CDL-A)')
ax2.plot(snr_jpg, ssim_jpg, 'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax2.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM vs SNR — CIFAR-10', fontsize=13)
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 27)
ax2.set_ylim(0, 1.05)

plt.suptitle('Image Semantic Communication vs Traditional — CIFAR-10', fontsize=13)
plt.tight_layout()
plt.savefig('checkpoints/image-jscc/eval/final_v3.png', dpi=150, bbox_inches='tight')
print("Saved: final_v3.png")

print(f"\n{'SNR':>6} {'ld128':>8} {'ld512':>8} {'CDL':>8} {'JPEG':>8}")
print("-" * 45)
for i in range(len(snr_128)):
    print(f"{snr_128[i]:>6} {psnr_128[i]:>8.2f} {psnr_512[i]:>8.2f} "
          f"{psnr_cdl[i]:>8.2f} {psnr_jpg[i]:>8.2f}")

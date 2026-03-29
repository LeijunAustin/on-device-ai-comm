import json, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

with open('checkpoints/image-jscc/eval/snr_recon_ld128.json') as f:
    awgn = json.load(f)
with open('checkpoints/image-jscc/eval/snr_recon_cdl.json') as f:
    cdl = json.load(f)
with open('checkpoints/image-jscc/eval/jpeg_baseline.json') as f:
    jpeg = json.load(f)

snr_a  = [d['ebno_db'] for d in awgn]
psnr_a = [d['psnr']    for d in awgn]
ssim_a = [d['ssim']    for d in awgn]

snr_c  = [d['ebno_db'] for d in cdl]
psnr_c = [d['psnr']    for d in cdl]
ssim_c = [d['ssim']    for d in cdl]

snr_j  = [d['ebno_db'] for d in jpeg]
psnr_j = [d['psnr'] if d['success_rate'] > 0.5 else 0 for d in jpeg]
ssim_j = [d['ssim'] if d['success_rate'] > 0.5 else 0 for d in jpeg]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PSNR
ax1.plot(snr_a, psnr_a, 'b-o',  lw=2, ms=7, label='Semantic CNN (AWGN)')
ax1.plot(snr_c, psnr_c, 'g-^',  lw=2, ms=7, label='Semantic CNN (CDL-A)')
ax1.plot(snr_j, psnr_j, 'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax1.axhline(y=21.64, color='gray', ls=':', lw=1.5, label='No channel upper bound')
ax1.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12)
ax1.set_title('PSNR vs SNR — CIFAR-10', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-7, 27)
ax1.set_ylim(0, 35)

# SSIM
ax2.plot(snr_a, ssim_a, 'b-o',  lw=2, ms=7, label='Semantic CNN (AWGN)')
ax2.plot(snr_c, ssim_c, 'g-^',  lw=2, ms=7, label='Semantic CNN (CDL-A)')
ax2.plot(snr_j, ssim_j, 'r--s', lw=2, ms=7, label='JPEG+LDPC (traditional)')
ax2.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_title('SSIM vs SNR — CIFAR-10', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-7, 27)
ax2.set_ylim(0, 1.05)

plt.suptitle('Image Semantic Communication: Semantic vs Traditional\n(CIFAR-10, CNN Encoder-Decoder)', fontsize=12)
plt.tight_layout()

out = 'checkpoints/image-jscc/eval/final_v2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

print("\n=== 结果汇总 ===")
print(f"{'SNR':>6} {'AWGN PSNR':>12} {'CDL PSNR':>11} {'JPEG PSNR':>11}")
print("-" * 45)
for i, snr in enumerate(snr_a):
    jp = psnr_j[i] if i < len(psnr_j) else 0
    print(f"{snr:>6} {psnr_a[i]:>12.2f} {psnr_c[i]:>11.2f} {jp:>11.2f}")

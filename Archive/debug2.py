import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_bgr = cv2.imread(str(Path.home() / "Downloads/puzzle2.jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
H, W = img_bgr.shape[:2]
print(f"Image: {W}×{H}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Show the full image cropped to left 200px (ruler region)
axes[0,0].imshow(img_rgb[:, :200])
axes[0,0].set_title("Left 200px (ruler area)")

# Vertical brightness profile of left strip
strip = img_rgb[:, 5:100, :]
gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
profile = gray.mean(axis=1)
axes[0,1].plot(profile, range(len(profile)))
axes[0,1].invert_yaxis()
axes[0,1].set_title("Vertical brightness profile (left strip)")
axes[0,1].set_xlabel("brightness"); axes[0,1].set_ylabel("y pixel")

# Show HSV channels of a sample orange patch
# Find rough orange region first
lo = np.array([0, 100, 100]); hi = np.array([30, 255, 255])
rough = cv2.inRange(img_hsv, lo, hi)
n, labels, stats, _ = cv2.connectedComponentsWithStats(rough, 8)
if n > 1:
    biggest = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
    bx, by, bw, bh = stats[biggest, :4]
    print(f"Largest orange-ish blob: {bw}×{bh} at ({bx},{by})")
    # Sample HSV in center of blob
    cx, cy = bx+bw//2, by+bh//2
    sample = img_hsv[cy-20:cy+20, cx-20:cx+20]
    print(f"  H: {sample[:,:,0].mean():.0f}  S: {sample[:,:,1].mean():.0f}  V: {sample[:,:,2].mean():.0f}")
    axes[0,2].imshow(img_rgb[by:by+bh, bx:bx+bw])
    axes[0,2].set_title(f"Orange blob ({bw}×{bh})")
else:
    print("No orange-ish blobs found!")
    axes[0,2].set_title("No orange found")

# H/S/V histograms of the whole image
for i, (ch, name) in enumerate(zip([0,1,2], ['Hue','Sat','Val'])):
    axes[1,i].hist(img_hsv[:,:,ch].flatten(), bins=60, color='steelblue')
    axes[1,i].set_title(f"{name} histogram")
    axes[1,i].set_xlabel("value"); axes[1,i].set_ylabel("count")

plt.tight_layout()
plt.savefig(Path.home() / "puzzle-piece/output/p2_debug.png", dpi=120)
print("Saved debug image")

# Print ruler profile stats
print(f"\nRuler strip profile: min={profile.min():.0f} max={profile.max():.0f} mean={profile.mean():.0f}")

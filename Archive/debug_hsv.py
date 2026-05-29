"""Sample HSV values at key spots in the image to inform thresholds."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_bgr = cv2.imread(str(Path.home() / "Downloads/puzzle.jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
H, W = img_bgr.shape[:2]

# Tight orange detection to find hole center
lo = np.array([5, 150, 120])
hi = np.array([25, 255, 255])
tight = cv2.inRange(img_hsv, lo, hi)
n, labels, stats, _ = cv2.connectedComponentsWithStats(tight, 8)
biggest = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
blob = (labels == biggest).astype(np.uint8) * 255
bx, by, bw, bh = cv2.boundingRect(blob)

# Show the 5 panels: original, tight mask, and HSV channels cropped to hole region
pad = 60
rx1, ry1 = max(0, bx - pad), max(0, by - pad)
rx2, ry2 = min(W, bx + bw + pad), min(H, by + bh + pad)

crop_rgb = img_rgb[ry1:ry2, rx1:rx2]
crop_hsv = img_hsv[ry1:ry2, rx1:rx2]
crop_blob = blob[ry1:ry2, rx1:rx2]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"Debug: hole ROI  tight bbox={bw}×{bh}px", fontsize=12)

axes[0,0].imshow(crop_rgb); axes[0,0].set_title("Original (cropped)")
axes[0,1].imshow(crop_blob, cmap="gray"); axes[0,1].set_title("Tight mask")
axes[0,2].imshow(crop_hsv[:,:,0], cmap="hsv", vmin=0, vmax=179)
axes[0,2].set_title("H channel")

axes[1,0].imshow(crop_hsv[:,:,1], cmap="gray"); axes[1,0].set_title("S channel")
axes[1,1].imshow(crop_hsv[:,:,2], cmap="gray"); axes[1,1].set_title("V channel")

# Scatter: H vs V for pixels in the hole ROI, coloured by whether tight mask caught them
h_vals = crop_hsv[:,:,0].flatten()
s_vals = crop_hsv[:,:,1].flatten()
v_vals = crop_hsv[:,:,2].flatten()
caught = crop_blob.flatten() > 0

# Only show pixels with moderate saturation (to exclude puzzle piece & background)
saturated = s_vals > 80
axes[1,2].scatter(h_vals[saturated & ~caught], v_vals[saturated & ~caught],
                  s=1, c="blue", alpha=0.3, label="missed (sat>80)")
axes[1,2].scatter(h_vals[saturated & caught], v_vals[saturated & caught],
                  s=1, c="red", alpha=0.3, label="caught")
axes[1,2].axhline(120, color="orange", lw=1.5, ls="--", label="V thresh=120")
axes[1,2].axvline(5, color="green", lw=1, ls=":", label="H lo=5")
axes[1,2].axvline(25, color="green", lw=1, ls=":", label="H hi=25")
axes[1,2].set_xlabel("Hue (0-179)"); axes[1,2].set_ylabel("Value (brightness)")
axes[1,2].legend(markerscale=5, fontsize=8)
axes[1,2].set_title("H vs V for saturated pixels")

for ax in axes.flat:
    ax.axis("off") if ax != axes[1,2] else None

plt.tight_layout()
out = Path.home() / "puzzle-piece/output/debug_hsv.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# Print stats for missed pixels
missed_h = h_vals[saturated & ~caught]
missed_v = v_vals[saturated & ~caught]
missed_s = s_vals[saturated & ~caught]
if len(missed_h):
    print(f"\nMissed pixels (sat>80): {len(missed_h)}")
    print(f"  H: min={missed_h.min()} max={missed_h.max()} median={np.median(missed_h):.0f}")
    print(f"  S: min={missed_s.min()} max={missed_s.max()} median={np.median(missed_s):.0f}")
    print(f"  V: min={missed_v.min()} max={missed_v.max()} median={np.median(missed_v):.0f}")

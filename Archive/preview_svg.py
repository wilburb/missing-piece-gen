"""Render piece.svg to a PNG for visual inspection."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_fill_holes
from pathlib import Path

OUT_DIR = Path.home() / "puzzle-piece/output"

# Re-run the same contour extraction so we can plot it cleanly
img_bgr = cv2.imread(str(Path.home() / "Downloads/puzzle.jpg"))
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_bgr.shape[:2]

# Load the final mask we saved
hole_mask = cv2.imread(str(OUT_DIR / "dbg_5_final.png"), cv2.IMREAD_GRAYSCALE)

# Extract contour
contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)

from scipy.ndimage import gaussian_filter1d
pts_raw = contour[:, 0, :].astype(float)
pts_smooth = np.column_stack([
    gaussian_filter1d(pts_raw[:, 0], sigma=3, mode="wrap"),
    gaussian_filter1d(pts_raw[:, 1], sigma=3, mode="wrap"),
])
step = max(1, len(pts_smooth) // 800)
pts = pts_smooth[::step]

# Bounding box
x, y, w, h = cv2.boundingRect(contour)

# Calibration
px_per_mm = 18.462

# ── Plot 1: contour overlaid on original image (cropped to hole area) ─────────
pad = 120
x1, y1 = max(0, x-pad), max(0, y-pad)
x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
crop = img_rgb[y1:y2, x1:x2]
pts_local = pts - np.array([x1, y1])

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.imshow(crop)
poly = plt.Polygon(pts_local, fill=False, edgecolor='lime', linewidth=2)
ax.add_patch(poly)
ax.set_title(f"Contour on image  ({w/px_per_mm:.1f} × {h/px_per_mm:.1f} mm)", fontsize=11)
ax.axis("off")

# ── Plot 2: standalone piece silhouette at ~true scale ─────────────────────────
ax = axes[1]
ax.set_aspect("equal")
ax.set_facecolor("#f0f0f0")

# Convert to mm, origin at (0,0)
pts_mm = (pts - np.array([x, y])) / px_per_mm
poly2 = plt.Polygon(pts_mm, closed=True, facecolor="white", edgecolor="#e05000",
                    linewidth=1.5)
ax.add_patch(poly2)

W_mm, H_mm = w / px_per_mm, h / px_per_mm
ax.set_xlim(-3, W_mm + 3)
ax.set_ylim(-3, H_mm + 3)

# Dimension annotations
ax.annotate("", xy=(W_mm, H_mm/2), xytext=(0, H_mm/2),
            arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(W_mm/2, H_mm/2 + 1.5, f"{W_mm:.1f} mm", ha="center", fontsize=9, color="#333")
ax.annotate("", xy=(-2, H_mm), xytext=(-2, 0),
            arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(-2, H_mm/2, f"{H_mm:.1f} mm", ha="center", fontsize=9, color="#333",
        rotation=90, va="center")

# 10 mm scale bar
ax.plot([0, 10], [-2, -2], color="#333", lw=2)
ax.text(5, -2.8, "10 mm", ha="center", fontsize=8, color="#333")

ax.set_title("Piece silhouette (true scale)", fontsize=11)
ax.set_xlabel("mm"); ax.set_ylabel("mm")

plt.tight_layout()
out = OUT_DIR / "piece_preview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

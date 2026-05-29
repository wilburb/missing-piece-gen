import cv2
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from pathlib import Path

IMAGE_PATH = Path.home() / "Downloads" / "puzzle2.jpg"
OUT_DIR = Path.home() / "puzzle-piece" / "output"
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Load ───────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(str(IMAGE_PATH))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
H, W = img_bgr.shape[:2]
print(f"Image size: {W}×{H} px")


# ── 2. Calibrate from vertical ruler ─────────────────────────────────────────
# puzzle2: ruler on left, mm side at x≈50-100, fine 1mm ticks ≈14-16 px apart.
# We detect all peaks loosely, then keep only gaps in the 1mm cluster
# (< 25 px), filtering out the larger 5mm/10mm jumps.
ruler_strip = img_rgb[:, 50:110, :]
ruler_gray  = cv2.cvtColor(ruler_strip, cv2.COLOR_RGB2GRAY)
profile = ruler_gray.mean(axis=1)

peaks, _ = find_peaks(profile, height=25, distance=6, prominence=3)
gaps = np.diff(peaks)
# 1mm ticks are the smallest consistent gap; filter out 5mm/10mm jumps
mm_gaps = gaps[gaps < 25]
px_per_mm = float(np.mean(mm_gaps)) if len(mm_gaps) >= 5 else 16.0
print(f"Detected {len(peaks)} peaks, {len(mm_gaps)} 1mm gaps → {px_per_mm:.3f} px/mm")
print(f"Calibration: {px_per_mm:.3f} px/mm")


# ── 3. Detect hole via white-piece complement ─────────────────────────────────
# Shadow on the right piece edge makes orange detection underestimate the hole.
# The piece backs are always bright white regardless of shadow.
# Strategy: label everything NOT white, then flood-fill from the orange centroid.
# The connected "not-white" region = the hole + all shadow strips, bounded
# cleanly by the white piece edges.

# Get orange centroid for the seed point only
lo_seed = np.array([0, 150, 150])
hi_seed = np.array([20, 255, 255])
seed_raw = cv2.inRange(img_hsv, lo_seed, hi_seed)
n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(seed_raw, 8)
big_s = st_s[1:, cv2.CC_STAT_AREA].argmax() + 1
m0 = cv2.moments((lbl_s == big_s).astype(np.uint8))
seed_x = int(m0["m10"] / m0["m00"])
seed_y = int(m0["m01"] / m0["m00"])
print(f"Seed: ({seed_x}, {seed_y})")

# Median-blur the image before thresholding: this smears the black text on
# the piece backs into the surrounding white, so pieces appear uniformly bright.
# ksize=41 covers text strokes that are up to ~20px thick.
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_med = cv2.medianBlur(gray, 41)

# After median blur the piece backs are uniformly bright; threshold on brightness
white_sealed = (gray_med > 160).astype(np.uint8) * 255

# "Not white" = hole + shadow (text is now absorbed into the piece regions)
not_white = (white_sealed == 0).astype(np.uint8)

# Connected component from seed = exactly the hole region
_, labels_nw = cv2.connectedComponents(not_white, connectivity=8)
hole_label = labels_nw[seed_y, seed_x]
blob = (labels_nw == hole_label).astype(np.uint8) * 255
cv2.imwrite(str(OUT_DIR / "p2_dbg_1_notwhite.png"), blob)

# Fill interior voids
blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255

# Small close for any residual seam noise
close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, close_k)
cv2.imwrite(str(OUT_DIR / "p2_dbg_2_closed.png"), blob)

# Fill again (close can create ring around a neighboring tab)
blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255

# Gaussian smooth the mask (sigma=12) → clean boundary before contour extraction
blob_f = gaussian_filter(blob.astype(np.float32) / 255.0, sigma=12)
hole_mask = (blob_f > 0.5).astype(np.uint8) * 255
cv2.imwrite(str(OUT_DIR / "p2_dbg_3_final.png"), hole_mask)


# ── 4. Fit parametric B-spline → evaluate as smooth polyline ──────────────────
contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_raw = max(contours, key=cv2.contourArea)
print(f"Contour points (raw): {len(contour_raw)}")

pts = contour_raw[:, 0, :].astype(float)

# splprep fits a smooth periodic B-spline.  s controls smoothing:
# larger s = smoother (loses fine detail), smaller = noisier.
# s ≈ N * 20 works well here; per=True closes the curve.
tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 20, per=True, k=3)

# Evaluate at 600 evenly-spaced parameter values → smooth closed polyline
u_fine = np.linspace(0, 1, 600, endpoint=False)
sx, sy = splev(u_fine, tck)
pts_smooth = np.column_stack([sx, sy])

# Also keep a cv2 contour for bounding-box / moments
contour = pts_smooth.astype(np.int32).reshape(-1, 1, 2)
print(f"Spline output points: {len(pts_smooth)}")

x, y, w, h = cv2.boundingRect(contour)
M = cv2.moments(contour)
cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
hole_px = int(cv2.contourArea(contour))

print(f"Hole bounding box: {w}×{h} px  →  {w/px_per_mm:.1f}×{h/px_per_mm:.1f} mm")


# ── 5. Visualisation ──────────────────────────────────────────────────────────
pad = 140
x1, y1 = max(0, x - pad), max(0, y - pad)
x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
crop = img_rgb[y1:y2, x1:x2]
pts_local = pts_smooth - np.array([x1, y1])

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Puzzle Piece Analysis (puzzle2.jpg — white back)", fontsize=13, fontweight="bold")

# Panel A: orange detection overlay
ax = axes[0]
ax.set_title("Orange Detection")
ax.imshow(img_rgb)
overlay = np.zeros_like(img_rgb)
overlay[hole_mask == 255] = [255, 80, 0]
ax.imshow(overlay, alpha=0.45)
ax.axis("off")

# Panel B: contour on cropped image
ax = axes[1]
ax.set_title(f"Contour  ({w/px_per_mm:.1f} × {h/px_per_mm:.1f} mm)")
ax.imshow(crop)
poly = plt.Polygon(pts_local, fill=False, edgecolor="lime", linewidth=2)
ax.add_patch(poly)
ax.axis("off")

# Panel C: standalone silhouette
ax = axes[2]
ax.set_aspect("equal")
ax.set_facecolor("#f0f0f0")
pts_mm = (pts_smooth - np.array([x, y])) / px_per_mm
W_mm, H_mm = w / px_per_mm, h / px_per_mm
poly2 = plt.Polygon(pts_mm, closed=True, facecolor="white",
                    edgecolor="#e05000", linewidth=1.5)
ax.add_patch(poly2)
ax.set_xlim(-3, W_mm + 3)
ax.set_ylim(-3, H_mm + 3)
ax.annotate("", xy=(W_mm, H_mm/2), xytext=(0, H_mm/2),
            arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(W_mm/2, H_mm/2 + 1.5, f"W={W_mm:.1f} mm", ha="center", fontsize=9)
ax.annotate("", xy=(-2, H_mm), xytext=(-2, 0),
            arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(-2.5, H_mm/2, f"H={H_mm:.1f} mm", ha="center", fontsize=9,
        rotation=90, va="center")
ax.plot([0, 10], [-2, -2], color="#333", lw=2)
ax.text(5, -2.8, "10 mm", ha="center", fontsize=8)
ax.set_title("Silhouette (true scale)")
ax.set_xlabel("mm"); ax.set_ylabel("mm")

plt.tight_layout()
fig.savefig(OUT_DIR / "p2_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'p2_analysis.png'}")


# ── 6. SVG export ─────────────────────────────────────────────────────────────
SVG_PX_PER_MM = 96 / 25.4
margin_mm = 10
margin = margin_mm * SVG_PX_PER_MM

pts_svg = (pts_smooth - np.array([x, y])) / px_per_mm * SVG_PX_PER_MM
svg_w = W_mm * SVG_PX_PER_MM + 2 * margin
svg_h = H_mm * SVG_PX_PER_MM + 2 * margin

dwg = svgwrite.Drawing(str(OUT_DIR / "p2_piece.svg"),
                       size=(f"{svg_w:.2f}px", f"{svg_h:.2f}px"), profile="full")
dwg.viewbox(0, 0, svg_w, svg_h)
dwg.add(dwg.rect(insert=(0, 0), size=(svg_w, svg_h), fill="#f8f8f8"))

# Build SVG path using cubic Bezier segments derived from the spline.
# At each sample point we get position + derivative; the control points are:
#   CP1 = P_i  + deriv_i  * dt/3
#   CP2 = P_{i+1} - deriv_{i+1} * dt/3
N_bez = 200   # number of Bezier segments (more = smoother approximation)
u_bez = np.linspace(0, 1, N_bez, endpoint=False)
dt = 1.0 / N_bez
bx,  by  = splev(u_bez,              tck)
dbx, dby = splev(u_bez,              tck, der=1)
bx1, by1 = splev((u_bez + dt) % 1,  tck)

def to_svg(px, py):
    sx2 = (px - x) / px_per_mm * SVG_PX_PER_MM + margin
    sy2 = (py - y) / px_per_mm * SVG_PX_PER_MM + margin
    return sx2, sy2

parts = [f"M {to_svg(bx[0], by[0])[0]:.3f},{to_svg(bx[0], by[0])[1]:.3f}"]
for i in range(N_bez):
    j = (i + 1) % N_bez
    # control point 1 (leaving i)
    cp1x = bx[i]  + dbx[i]  * dt / 3
    cp1y = by[i]  + dby[i]  * dt / 3
    # evaluate derivative at next point
    dbx2, dby2 = splev(u_bez[j], tck, der=1)
    # control point 2 (arriving at j)
    cp2x = bx[j]  - dbx2 * dt / 3
    cp2y = by[j]  - dby2 * dt / 3
    ex, ey   = to_svg(bx[j],  by[j])
    c1x, c1y = to_svg(cp1x, cp1y)
    c2x, c2y = to_svg(cp2x, cp2y)
    parts.append(f"C {c1x:.3f},{c1y:.3f} {c2x:.3f},{c2y:.3f} {ex:.3f},{ey:.3f}")
parts.append("Z")
path_d = " ".join(parts)
dwg.add(dwg.path(d=path_d, fill="none", stroke="#e05000", stroke_width=0.5))

bb_w, bb_h = W_mm * SVG_PX_PER_MM, H_mm * SVG_PX_PER_MM
dwg.add(dwg.rect(insert=(margin, margin), size=(bb_w, bb_h),
                 fill="none", stroke="#aaa", stroke_width=0.3,
                 stroke_dasharray="4,3"))
txt = "font-size:8px; font-family:monospace; fill:#333;"
dwg.add(dwg.text(f"W = {W_mm:.1f} mm",
                 insert=(margin + bb_w/2, margin - 4), style=txt,
                 text_anchor="middle"))
dwg.add(dwg.text(f"H = {H_mm:.1f} mm",
                 insert=(margin - 4, margin + bb_h/2), style=txt,
                 text_anchor="middle",
                 transform=f"rotate(-90,{margin-4},{margin+bb_h/2})"))
bar = 10 * SVG_PX_PER_MM
bar_y = margin + bb_h + margin * 0.6
dwg.add(dwg.line(start=(margin, bar_y), end=(margin + bar, bar_y),
                 stroke="#333", stroke_width=1))
dwg.add(dwg.text("10 mm", insert=(margin + bar/2, bar_y + 9),
                 style=txt, text_anchor="middle"))
dwg.save()
print(f"Saved: {OUT_DIR / 'p2_piece.svg'}")

print(f"\n── Summary ──────────────────────────────────────────────────────────")
print(f"  Calibration : {px_per_mm:.3f} px/mm")
print(f"  Hole width  : {w/px_per_mm:.1f} mm")
print(f"  Hole height : {h/px_per_mm:.1f} mm")
print(f"  Hole area   : {hole_px/px_per_mm**2:.1f} mm²")
print(f"─────────────────────────────────────────────────────────────────────")

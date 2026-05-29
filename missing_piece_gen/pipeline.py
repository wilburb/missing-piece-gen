"""Pipeline stages for missing piece generation.

Implements the Archive (analyze.py) approach:
  1. load_image      — cv2 load + optional ruler calibration
  2. detect_missing_region — orange seed → medianBlur → not-white complement
                             → connected component → B-spline hole contour
  3. generate_3d_model — wrap outline in MissingPieceShape
  4. write_output      — extrude and export via model_gen
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks

from .models import MissingPieceShape
from .errors import DetectionError
from . import model_gen

# Tight HSV range for the orange seed point (same as Archive analyze.py)
_SEED_HSV_LOW  = np.array([0,  150, 150], dtype=np.uint8)
_SEED_HSV_HIGH = np.array([20, 255, 255], dtype=np.uint8)

# Gray brightness above which a pixel is treated as a white puzzle back
_WHITE_THRESHOLD = 160

# Default px/mm when no ruler is found (Archive default)
_DEFAULT_PX_PER_MM = 16.0


def load_image(image_path: str) -> dict:
    """Load the puzzle image and calibrate scale from ruler if present.

    Returns a dict with keys:
      image     – BGR ndarray
      gray      – grayscale ndarray
      px_per_mm – float, pixels per mm (from ruler or default 16)
      width, height – image dimensions
    """
    img = cv2.imread(image_path)
    if img is None:
        raise DetectionError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px_per_mm = _calibrate_from_ruler(gray) or _DEFAULT_PX_PER_MM

    return {
        "image":     img,
        "gray":      gray,
        "px_per_mm": px_per_mm,
        "width":     img.shape[1],
        "height":    img.shape[0],
    }


def _calibrate_from_ruler(gray: np.ndarray) -> float | None:
    """Detect px/mm from a vertical mm-ruler strip at x=50-110 (Archive approach).

    Matches the calibration logic in analyze.py:  find_peaks with
    height=25, distance=6, prominence=3 on the horizontal mean profile of the
    left strip; keep only sub-25px gaps (1mm ticks); require >=5 gaps.
    """
    if gray.shape[1] < 111:
        return None

    strip   = gray[:, 50:110]
    profile = strip.mean(axis=1).astype(np.float32)

    peaks, _ = find_peaks(profile, height=25, distance=6, prominence=3)
    if len(peaks) < 2:
        return None

    gaps    = np.diff(peaks.astype(float))
    mm_gaps = gaps[gaps < 25]

    if len(mm_gaps) < 5:
        return None

    return float(np.mean(mm_gaps))


def detect_missing_region(image_data: dict) -> dict:
    """Detect the missing-piece hole using the Archive not-white complement strategy.

    Algorithm (mirrors analyze.py steps 3-4):

    1. Find the orange region centroid (tight HSV) — this is the seed point
       that anchors the connected-component search inside the hole.
    2. Apply medianBlur(gray, ksize=41).  The large kernel absorbs text/print
       on white puzzle backs so the entire piece face appears uniformly bright.
    3. Threshold gray_med > 160 → white_sealed.  Invert → not_white.
    4. connectedComponents(not_white, 8).  The label at the seed = hole region.
       (Shadows on piece edges are also not-white but they are bounded by the
       clean white piece faces, so they merge into the hole label cleanly.)
    5. binary_fill_holes → morph_close(9x9) → binary_fill_holes →
       gaussian_filter(sigma=12) → re-threshold.
    6. findContours → largest contour → splprep B-spline (s=N*20, per=True)
       evaluated at 600 evenly-spaced parameter values.
    7. Convert to mm-space relative to the contour bounding box.

    Returns a dict with keys:
      outline_mm       – Nx2 closed polygon in mm (origin = contour top-left)
      width_mm         – bounding-box width in mm
      height_mm        – bounding-box height in mm
      px_per_mm        – scale used
      hole_contour_px  – raw B-spline points in image pixel coords
    """
    img      = image_data["image"]
    gray     = image_data["gray"]
    px_per_mm = image_data["px_per_mm"]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Step 1: orange seed centroid ─────────────────────────────────────────
    seed_raw = cv2.inRange(img_hsv, _SEED_HSV_LOW, _SEED_HSV_HIGH)
    n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(seed_raw, 8)
    if n_s <= 1:
        raise DetectionError(
            "No orange region found. Place orange paper in the missing slot "
            "so the algorithm can locate the hole."
        )
    big_s = 1 + int(np.argmax(st_s[1:, cv2.CC_STAT_AREA]))
    m0 = cv2.moments((lbl_s == big_s).astype(np.uint8))
    if m0["m00"] == 0:
        raise DetectionError("Orange region has zero area.")
    seed_x = int(m0["m10"] / m0["m00"])
    seed_y = int(m0["m01"] / m0["m00"])

    # ── Step 2-3: medianBlur seals piece-back text; threshold → white / not-white
    gray_med    = cv2.medianBlur(gray, 41)
    white_sealed = (gray_med > _WHITE_THRESHOLD).astype(np.uint8) * 255
    not_white   = cv2.bitwise_not(white_sealed)

    # ── Step 4: connected component from seed = hole region ──────────────────
    _, labels_nw = cv2.connectedComponents(not_white, connectivity=8)
    hole_label = labels_nw[seed_y, seed_x]
    if hole_label == 0:
        raise DetectionError(
            "The orange seed point lands inside a white region after median blur. "
            "The orange paper may be too small or too bright."
        )
    blob = (labels_nw == hole_label).astype(np.uint8) * 255

    # ── Step 5: mask cleanup ─────────────────────────────────────────────────
    blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, close_k)
    blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
    blob_f = gaussian_filter(blob.astype(np.float32) / 255.0, sigma=12)
    hole_mask = (blob_f > 0.5).astype(np.uint8) * 255

    # ── Step 6: contour + B-spline ───────────────────────────────────────────
    contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise DetectionError("Could not extract hole contour after mask cleanup.")

    pts_raw = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    n = len(pts_raw)
    if n < 10:
        raise DetectionError(
            f"Hole contour has only {n} points; image may be too low-resolution."
        )

    tck, _ = splprep([pts_raw[:, 0], pts_raw[:, 1]], s=n * 20, per=True, k=3)
    u_fine = np.linspace(0, 1, 600, endpoint=False)
    sx, sy = splev(u_fine, tck)
    pts_smooth = np.column_stack([sx, sy])

    # ── Step 7: convert to mm, origin at contour bounding-box top-left ───────
    x_min, y_min = pts_smooth.min(axis=0)
    x_max, y_max = pts_smooth.max(axis=0)
    width_px  = x_max - x_min
    height_px = y_max - y_min

    mm_per_px  = 1.0 / px_per_mm
    outline_mm = (pts_smooth - np.array([x_min, y_min])) * mm_per_px
    outline_mm = np.vstack([outline_mm, outline_mm[0]])  # close the polygon

    return {
        "outline_mm":      outline_mm,
        "width_mm":        width_px  * mm_per_px,
        "height_mm":       height_px * mm_per_px,
        "px_per_mm":       px_per_mm,
        "hole_contour_px": pts_smooth,
    }


def generate_3d_model(region_data: dict, output_format: str) -> dict:
    """Wrap the detected outline in a MissingPieceShape ready for extrusion."""
    shape = MissingPieceShape(
        outline=region_data["outline_mm"],
        width_mm=region_data["width_mm"],
        height_mm=region_data["height_mm"],
        pixel_to_mm_scale=1.0 / region_data["px_per_mm"],
    )
    return {"shape": shape, "format": output_format}


def write_output(model_data: dict, output_path: str) -> None:
    """Extrude the 2D outline into a watertight 3D solid and write to disk."""
    model_gen.generate(
        shape=model_data["shape"],
        output_path=output_path,
        format=model_data["format"],
    )

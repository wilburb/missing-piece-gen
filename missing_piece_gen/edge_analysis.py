"""Stage 2: Extract edge profiles from puzzle piece regions."""
import numpy as np
import cv2
from .models import PieceRegion, EdgeProfile, EdgeType, TabGeometry
from .errors import EdgeExtractionError

# Pixels of deviation required to classify as TAB or BLANK
_DEVIATION_THRESHOLD = 5.0

# Fraction of the crop to use as the edge region of interest
_EDGE_ROI_FRACTION = 0.30


def extract_edges(piece: PieceRegion) -> list[EdgeProfile]:
    """
    Extract contours and classify edge types for all inward-facing edges.

    Args:
        piece: A PieceRegion from stage 1 segmentation.

    Returns:
        List of EdgeProfile objects, one per inward-facing edge.

    Raises:
        EdgeExtractionError: If edge extraction fails for this piece.
    """
    if piece.crop is None or piece.crop.size == 0:
        raise EdgeExtractionError(
            f"Piece {piece.piece_id}: crop is empty or None"
        )

    profiles: list[EdgeProfile] = []
    for direction in piece.inward_edges:
        profile = _extract_single_edge(piece, direction)
        profiles.append(profile)
    return profiles


def _extract_single_edge(piece: PieceRegion, direction: str) -> EdgeProfile:
    """Extract and classify one edge from the piece crop."""
    crop = piece.crop
    h, w = crop.shape[:2]

    # --- 1. Isolate edge region of interest ---
    roi, roi_offset = _get_edge_roi(crop, direction, h, w)

    # Guard: if the ROI is too small for the GaussianBlur (5,5) kernel, return
    # a flat edge immediately rather than letting OpenCV raise a C++ exception.
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        return EdgeProfile(
            direction=direction,
            contour=np.empty((0, 2), dtype=np.float32),
            edge_type=EdgeType.FLAT,
            tab_geometry=None,
        )

    # --- 2. Build piece mask ---
    # Use .copy() to guarantee a C-contiguous array before passing to OpenCV.
    gray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
    # Use Otsu thresholding so the piece/background separation adapts to the
    # actual intensity distribution in the ROI.  A fixed threshold of 10 made
    # the entire mask white for any real photo (puzzle pixels are 100-255),
    # causing findContours to trace the ROI border and yielding zero deviation.
    #
    # On Windows, certain row widths trigger OpenCV's SIMD path to raise an
    # "Unknown C++ exception" inside GaussianBlur or threshold.  Wrap the entire
    # block through findContours in a try/except and return FLAT on failure.
    try:
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- 3. Find contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except Exception:
        return EdgeProfile(
            direction=direction,
            contour=np.empty((0, 2), dtype=np.float32),
            edge_type=EdgeType.FLAT,
            tab_geometry=None,
        )

    if not contours:
        # Can't extract — return a flat edge with empty contour
        return EdgeProfile(
            direction=direction,
            contour=np.empty((0, 2), dtype=np.float32),
            edge_type=EdgeType.FLAT,
            tab_geometry=None,
        )

    # Use the largest contour
    contour_raw = max(contours, key=cv2.contourArea)

    # Smooth with approxPolyDP (low epsilon so we keep detail)
    epsilon = 0.005 * cv2.arcLength(contour_raw, True)
    contour_smooth = cv2.approxPolyDP(contour_raw, epsilon, True)

    # Reshape to (N, 2) and offset into crop coordinates
    pts = contour_smooth.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += roi_offset[0]  # x offset
    pts[:, 1] += roi_offset[1]  # y offset

    # --- 4. Extract edge-side contour points and compute baseline ---
    edge_pts = _filter_edge_points(pts, direction, h, w)

    if len(edge_pts) < 2:
        return EdgeProfile(
            direction=direction,
            contour=pts,
            edge_type=EdgeType.FLAT,
            tab_geometry=None,
        )

    # Sort points along the baseline axis
    edge_pts = _sort_along_baseline(edge_pts, direction)

    baseline_start = edge_pts[0]
    baseline_end = edge_pts[-1]

    # --- 5. Compute signed deviations from the baseline ---
    deviations = _signed_deviations(edge_pts, baseline_start, baseline_end, direction)

    # --- 6. Classify ---
    edge_type, tab_geometry = _classify(deviations, edge_pts, direction)

    return EdgeProfile(
        direction=direction,
        contour=edge_pts,
        edge_type=edge_type,
        tab_geometry=tab_geometry,
    )


def _get_edge_roi(
    crop: np.ndarray, direction: str, h: int, w: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Return (roi_image, (x_offset, y_offset)) for the given edge direction.
    The ROI covers the ~30% of the crop closest to the named edge.
    """
    frac = _EDGE_ROI_FRACTION
    if direction == "top":
        roi_h = max(1, int(h * frac))
        return crop[:roi_h, :], (0, 0)
    elif direction == "bottom":
        roi_h = max(1, int(h * frac))
        y0 = h - roi_h
        return crop[y0:, :], (0, y0)
    elif direction == "left":
        roi_w = max(1, int(w * frac))
        return crop[:, :roi_w], (0, 0)
    elif direction == "right":
        roi_w = max(1, int(w * frac))
        x0 = w - roi_w
        return crop[:, x0:], (x0, 0)
    else:
        # Unknown direction — use full crop
        return crop, (0, 0)


def _filter_edge_points(
    pts: np.ndarray, direction: str, h: int, w: int
) -> np.ndarray:
    """
    Keep only contour points that lie near the edge of interest.
    'Near' is defined as the outermost 40% of the crop along the
    perpendicular axis.
    """
    # Keep only points in the outer 20% of the crop (inside the 30% ROI band).
    # The old 40% threshold was wider than the ROI, so nothing was filtered.
    frac = 0.20
    if direction == "top":
        threshold = h * frac
        mask = pts[:, 1] <= threshold
    elif direction == "bottom":
        threshold = h * (1.0 - frac)
        mask = pts[:, 1] >= threshold
    elif direction == "left":
        threshold = w * frac
        mask = pts[:, 0] <= threshold
    elif direction == "right":
        threshold = w * (1.0 - frac)
        mask = pts[:, 0] >= threshold
    else:
        return pts

    filtered = pts[mask]
    return filtered if len(filtered) >= 2 else pts


def _sort_along_baseline(pts: np.ndarray, direction: str) -> np.ndarray:
    """Sort points along the primary axis of the given edge direction."""
    if direction in ("top", "bottom"):
        idx = np.argsort(pts[:, 0])  # sort by x
    else:
        idx = np.argsort(pts[:, 1])  # sort by y
    return pts[idx]


def _signed_deviations(
    pts: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    Compute signed perpendicular deviations of each point from the
    baseline (start -> end).

    For "top"/"bottom" edges the baseline is roughly horizontal; the
    sign convention is:
      - top edge: positive deviation = point protrudes upward (into slot)
      - bottom edge: positive deviation = point protrudes downward (into slot)
      - left edge: positive deviation = point protrudes leftward (into slot)
      - right edge: positive deviation = point protrudes rightward (into slot)
    """
    if np.allclose(start, end):
        return np.zeros(len(pts))

    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    # Perpendicular unit vector (rotated 90° CCW)
    perp = np.array([-line_unit[1], line_unit[0]])

    rel = pts - start
    signed = rel @ perp  # dot product of each point with perp

    # Flip sign so "protrusion into slot" is positive
    if direction == "top":
        signed = -signed   # image y increases downward; tab protrudes upward (smaller y)
    elif direction == "left":
        signed = -signed   # tab protrudes leftward (smaller x)

    return signed


def _classify(
    deviations: np.ndarray,
    pts: np.ndarray,
    direction: str,
) -> tuple[EdgeType, TabGeometry | None]:
    """Classify the edge and compute TabGeometry if applicable."""
    if len(deviations) == 0:
        return EdgeType.FLAT, None

    max_dev = float(np.max(deviations))
    min_dev = float(np.min(deviations))

    if max_dev > _DEVIATION_THRESHOLD:
        edge_type = EdgeType.TAB
        peak_dev = max_dev
        feature_mask = deviations > (_DEVIATION_THRESHOLD / 2.0)
    elif min_dev < -_DEVIATION_THRESHOLD:
        edge_type = EdgeType.BLANK
        peak_dev = abs(min_dev)
        feature_mask = deviations < -(_DEVIATION_THRESHOLD / 2.0)
    else:
        return EdgeType.FLAT, None

    # Compute TabGeometry
    tab_geom = _compute_tab_geometry(deviations, pts, feature_mask, peak_dev, direction)
    return edge_type, tab_geom


def _compute_tab_geometry(
    deviations: np.ndarray,
    pts: np.ndarray,
    feature_mask: np.ndarray,
    peak_dev: float,
    direction: str,
) -> TabGeometry:
    """Compute normalized position, width, and depth of the tab/blank feature."""
    n = len(pts)
    if n < 2:
        return TabGeometry(position=0.5, width=0.0, depth=peak_dev)

    # Baseline axis positions
    if direction in ("top", "bottom"):
        axis_vals = pts[:, 0]
    else:
        axis_vals = pts[:, 1]

    total_span = float(axis_vals[-1] - axis_vals[0])
    if total_span <= 0:
        return TabGeometry(position=0.5, width=0.0, depth=peak_dev)

    feature_indices = np.where(feature_mask)[0]
    if len(feature_indices) == 0:
        return TabGeometry(position=0.5, width=0.0, depth=peak_dev)

    feature_start = axis_vals[feature_indices[0]]
    feature_end = axis_vals[feature_indices[-1]]
    feature_center = (feature_start + feature_end) / 2.0
    feature_width = float(feature_end - feature_start)

    position = float((feature_center - axis_vals[0]) / total_span)
    position = max(0.0, min(1.0, position))

    return TabGeometry(
        position=position,
        width=feature_width,
        depth=peak_dev,
    )

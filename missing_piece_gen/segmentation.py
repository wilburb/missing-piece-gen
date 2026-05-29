"""Stage 1: Detect and segment surrounding puzzle pieces from a photo."""
import numpy as np
import cv2
from .models import PieceRegion
from .errors import DetectionError

# Laplacian variance below this threshold indicates a blurry image.
_BLUR_THRESHOLD = 50.0

# Minimum contour area (pixels^2) to avoid noise blobs.
# 500 is too small for real photos — countertop specks, dust marks, etc. easily
# exceed 500 px².  Piece detection also applies a size-relative lower bound.
_MIN_CONTOUR_AREA = 2000

# Maximum contour area fraction of the total image area.
_MAX_CONTOUR_AREA_FRACTION = 0.70

# How many intensity levels above the estimated background to threshold for pieces.
_PIECE_INTENSITY_OFFSET = 3

# ---------------------------------------------------------------------------
# Prusa orange backdrop colour (OpenCV HSV, H: 0–180, S/V: 0–255)
# Prusa orange ≈ #FA6831 → H 18°, S 80%, V 98%
# Tight range: white puzzle backs have low S so they won't bleed into the
# orange mask.  The --orange-ref calibration path still overrides these.
# ---------------------------------------------------------------------------
_ORANGE_HSV_LOW  = np.array([ 0, 150, 150], dtype=np.uint8)
_ORANGE_HSV_HIGH = np.array([20, 255, 255], dtype=np.uint8)

# When orange covers more than this fraction of the total image area it is
# treated as a full backdrop rather than a small interior slot marker.
# A full backdrop WILL touch all four image edges — the touches_edge guard is
# intentionally relaxed for regions above this size threshold.
_ORANGE_BACKDROP_FRACTION = 0.10


def _fit_bspline_contour(pts: np.ndarray, n_out: int = 600) -> np.ndarray:
    """Fit a closed periodic B-spline to a contour and return n_out smooth points.

    Archive approach: splprep with s=N*20, per=True, evaluated at n_out evenly-spaced
    parameter values.  Falls back to the original points on any error (too few points,
    degenerate curve, etc.).
    """
    from scipy.interpolate import splprep, splev
    n = len(pts)
    if n < 10:
        return pts
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=n * 20, per=True, k=3)
        u_fine = np.linspace(0, 1, min(n_out, max(n, 60)), endpoint=False)
        sx, sy = splev(u_fine, tck)
        return np.column_stack([sx, sy])
    except Exception:
        return pts


def _check_blur(gray: np.ndarray) -> None:
    """Raise DetectionError if the image is too blurry."""
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < _BLUR_THRESHOLD:
        raise DetectionError(
            f"Image is too blurry (Laplacian variance {variance:.1f} < {_BLUR_THRESHOLD}). "
            "Please provide a sharper photo."
        )


def calibrate_orange_hsv(
    ref_image: np.ndarray,
    h_margin: int = 10,
    s_margin: int = 40,
    v_margin: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Auto-calibrate the HSV orange range from a reference image.

    Converts *ref_image* (BGR) to HSV, computes the 5th and 95th percentile
    for each channel across all pixels, then expands by the given margins and
    clamps to the valid OpenCV HSV range (H: 0–180, S/V: 0–255).

    Args:
        ref_image:  BGR image containing only the orange backdrop.
        h_margin:   Expansion margin for the Hue channel.
        s_margin:   Expansion margin for the Saturation channel.
        v_margin:   Expansion margin for the Value channel.

    Returns:
        (low, high) as uint8 np.arrays of shape (3,).
    """
    hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    margins = np.array([h_margin, s_margin, v_margin], dtype=np.float32)
    maxvals = np.array([180.0, 255.0, 255.0], dtype=np.float32)

    p5  = np.percentile(pixels, 5,  axis=0)
    p95 = np.percentile(pixels, 95, axis=0)

    low  = np.clip(p5  - margins, 0.0, maxvals).astype(np.uint8)
    high = np.clip(p95 + margins, 0.0, maxvals).astype(np.uint8)

    return low, high


def calibrate_from_ruler(gray: np.ndarray) -> float | None:
    """Detect a mm ruler in the image and return px/mm scale.

    Scans a vertical strip at x=50–110px, collapses horizontally to a
    1D brightness profile, finds tick-mark peaks, filters to gaps < 25px
    (the 1mm ticks), and returns the mean gap (px/mm).

    Returns None if fewer than 3 usable gaps are found.
    """
    from scipy.signal import find_peaks

    if gray.shape[1] < 111:
        return None

    strip = gray[:, 50:110]
    profile = strip.mean(axis=1).astype(np.float32)

    peaks, _ = find_peaks(profile, distance=5)
    if len(peaks) < 2:
        return None

    gaps = np.diff(peaks.astype(float))
    mm_gaps = gaps[gaps < 25]

    if len(mm_gaps) < 3:
        return None

    return float(mm_gaps.mean())


def _find_orange_backdrop(
    hsv: np.ndarray,
    img_w: int,
    img_h: int,
    gray: np.ndarray | None = None,
    hsv_low: np.ndarray | None = None,
    hsv_high: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool, str, np.ndarray | None]:
    """Detect Prusa orange in the image and determine how it should be used.

    Two modes are distinguished:

    * **Backdrop mode** – the orange region is dominant (covers more than
      ``_ORANGE_BACKDROP_FRACTION`` of the image area).  In this case the
      orange spans the whole background and its bounding rect will touch the
      image borders.  The function applies a multi-step mask cleaning pipeline
      (largest-blob selection, fill_holes, morph-close, Gaussian blur), extracts
      and smooths the outer contour, and sets ``is_backdrop=True``.  The caller
      should then find pieces as the non-orange blobs within the frame.  The
      cleaned mask is returned as the fourth element so it can be reused by
      ``_find_piece_contours_by_non_orange``.

    * **Slot-marker mode** – a smaller orange region that does NOT touch the
      border, intended as a colour-coded slot marker placed inside the missing
      slot.  ``is_backdrop=False``.  The fourth return value is ``None``.

    Args:
        hsv:      HSV image.
        img_w:    Image width in pixels.
        img_h:    Image height in pixels.
        hsv_low:  Lower HSV bound (uint8 array of shape (3,)).  Defaults to
                  ``_ORANGE_HSV_LOW`` when None.
        hsv_high: Upper HSV bound (uint8 array of shape (3,)).  Defaults to
                  ``_ORANGE_HSV_HIGH`` when None.

    Returns:
        (contour_or_None, is_backdrop, status_message, cleaned_mask_or_None)
    """
    from scipy.ndimage import binary_fill_holes, gaussian_filter

    if hsv_low is None:
        hsv_low = _ORANGE_HSV_LOW
    if hsv_high is None:
        hsv_high = _ORANGE_HSV_HIGH

    img_area = float(img_h * img_w)

    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # Morphological cleanup: close small gaps, remove small specks.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

    orange_px = int(np.count_nonzero(mask))
    orange_fraction = orange_px / img_area

    # --- Check if orange fraction qualifies as a backdrop ---
    if orange_fraction >= _ORANGE_BACKDROP_FRACTION:
        # Step A: keep largest connected blob
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest, np.uint8(255), np.uint8(0))

        # Save a pre-fill-holes copy for piece detection.  binary_fill_holes fills
        # any non-orange region that is fully enclosed by orange — including puzzle
        # pieces on the surface — so piece detection must use the mask before holes
        # are filled.  The filled mask is used only for contour boundary extraction.
        piece_detection_mask = mask.copy()

        # Step B: fill_holes pass 1 (fills dark spots/reflections inside orange region)
        mask = binary_fill_holes(mask.astype(bool)).astype(np.uint8) * 255

        # Step C: morph close (seals thin shadow strips at piece edges)
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)

        # Step D: fill_holes pass 2 (fills topological ring artifact from morph-close around tabs)
        mask = binary_fill_holes(mask.astype(bool)).astype(np.uint8) * 255

        # Step E: Gaussian smooth mask + re-threshold (Archive approach: sigma=12 on
        # normalised float mask produces a much cleaner boundary than cv2.GaussianBlur).
        blob_f = gaussian_filter(mask.astype(np.float32) / 255.0, sigma=12)
        mask = (blob_f > 0.5).astype(np.uint8) * 255

        # Extract contour from cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, False, "  [orange] No backdrop contour after cleanup.", None

        # B-spline smoothing (Archive approach) produces a far smoother closed contour
        # than the previous gaussian_filter1d + subsample strategy.
        pts_raw = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        pts = _fit_bspline_contour(pts_raw, n_out=600)

        # Convert back to contour format for cv2 compatibility
        best = pts.astype(np.int32).reshape(-1, 1, 2)
        backdrop_best_area = float(cv2.contourArea(best))
        bx, by, bw, bh = cv2.boundingRect(best)
        msg = (
            f"  [orange] Backdrop detected: "
            f"bbox=({bx}, {by}, {bw}x{bh} px)  area={backdrop_best_area:.0f} px2  "
            f"({100 * backdrop_best_area / img_area:.1f}% of image).\n"
            "         Piece detection will use non-orange blob strategy."
        )
        return best, True, msg, piece_detection_mask

    # --- Pass 2: look for a smaller interior slot marker (must not touch border) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_best: np.ndarray | None = None
    marker_best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_CONTOUR_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        touches_edge = (
            bx <= 1 or by <= 1
            or bx + bw >= img_w - 1
            or by + bh >= img_h - 1
        )
        if touches_edge:
            continue
        if area > marker_best_area:
            marker_best_area = area
            marker_best = c

    if marker_best is None:
        return None, False, (
            f"  [orange] No orange region found "
            f"(total orange pixels: {orange_px}; fraction={orange_fraction:.3f}; "
            f"expected HSV {hsv_low.tolist()} - {hsv_high.tolist()}). "
            "Falling back to dark-region slot detection."
        ), None

    # Archive approach: when a grayscale image is available, refine the hole
    # boundary using medianBlur + not-white complement + seed-connected-component.
    # This handles shadows on piece edges that cause direct orange tracking to
    # underestimate the hole area (the key insight from analyze.py in Archive/).
    if gray is not None:
        bx_m, by_m, bw_m, bh_m = cv2.boundingRect(marker_best)
        seed_x = bx_m + bw_m // 2
        seed_y = by_m + bh_m // 2
        # ksize=41 covers text strokes up to ~20px thick on white puzzle backs.
        gray_med = cv2.medianBlur(gray, 41)
        white_sealed = (gray_med > 160).astype(np.uint8) * 255
        not_white = cv2.bitwise_not(white_sealed)
        _, labels_nw = cv2.connectedComponents(not_white, connectivity=8)
        hole_label = labels_nw[seed_y, seed_x]
        if hole_label != 0:
            blob = (labels_nw == hole_label).astype(np.uint8) * 255
            blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, close_k)
            blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
            blob_f = gaussian_filter(blob.astype(np.float32) / 255.0, sigma=12)
            refined = (blob_f > 0.5).astype(np.uint8) * 255
            refined_contours, _ = cv2.findContours(
                refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if refined_contours:
                pts_raw = max(refined_contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
                pts = _fit_bspline_contour(pts_raw, n_out=600)
                marker_best = pts.astype(np.int32).reshape(-1, 1, 2)

    # Report mean HSV of the detected marker so the user can refine the range.
    sx, sy, sw, sh = cv2.boundingRect(marker_best)
    slot_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(slot_mask, [marker_best], -1, 255, cv2.FILLED)
    mean_vals = cv2.mean(hsv, mask=slot_mask)
    mean_h, mean_s, mean_v = mean_vals[0], mean_vals[1], mean_vals[2]

    msg = (
        f"  [orange] Slot marker detected: "
        f"bbox=({sx}, {sy}, {sw}x{sh} px)  area={marker_best_area:.0f} px2\n"
        f"         Detected mean HSV=({mean_h:.0f}, {mean_s:.0f}, {mean_v:.0f})  "
        f"[range H {hsv_low[0]}-{hsv_high[0]}, "
        f"S {hsv_low[1]}-{hsv_high[1]}, "
        f"V {hsv_low[2]}-{hsv_high[2]}]"
    )
    return marker_best, False, msg, None


def _find_slot_contour(
    gray: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """Fallback: find the slot as the largest interior dark region (Otsu)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv_binary = cv2.bitwise_not(binary)
    dark_contours, _ = cv2.findContours(inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slot_contour = None
    slot_area = 0.0
    for c in dark_contours:
        area = cv2.contourArea(c)
        if area < _MIN_CONTOUR_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        touches_edge = (
            bx <= 1 or by <= 1
            or bx + bw >= img_w - 1
            or by + bh >= img_h - 1
        )
        if touches_edge:
            continue
        if area > slot_area:
            slot_area = area
            slot_contour = c

    return slot_contour


def _inward_edges_for_piece(
    piece_box: tuple[int, int, int, int],
    slot_box: tuple[int, int, int, int],
) -> list[str]:
    """Determine which edges of *piece_box* face the missing slot."""
    px, py, pw, ph = piece_box
    sx, sy, sw, sh = slot_box

    pc_x = px + pw / 2
    pc_y = py + ph / 2
    sc_x = sx + sw / 2
    sc_y = sy + sh / 2

    dx = sc_x - pc_x
    dy = sc_y - pc_y

    edges: list[str] = []
    threshold = 0.4 * max(abs(dx), abs(dy), 1.0)

    if dx > threshold:
        edges.append("right")
    if dx < -threshold:
        edges.append("left")
    if dy > threshold:
        edges.append("bottom")
    if dy < -threshold:
        edges.append("top")

    if not edges:
        edges = ["right"]

    return edges


def _estimate_background_intensity(gray: np.ndarray) -> int:
    """Estimate the dominant background intensity (modal brightness)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist[:50] = 0
    return int(np.argmax(hist))


def _find_piece_contours(
    gray: np.ndarray,
    img_w: int,
    img_h: int,
    img_area: float,
    slot_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Find contours that correspond to puzzle pieces (intensity-threshold strategy).

    Excludes the slot region (if slot_mask provided) and uses a size-relative
    minimum area to reject countertop specks and small noise blobs.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bg_intensity = _estimate_background_intensity(blurred)

    threshold_value = max(0, bg_intensity + _PIECE_INTENSITY_OFFSET)
    _, piece_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Blank out the slot region so the orange/dark slot isn't detected as a piece.
    if slot_mask is not None:
        piece_mask = cv2.bitwise_and(piece_mask, cv2.bitwise_not(slot_mask))

    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use a size-relative lower bound: at least 0.05% of image area.
    # This eliminates countertop specks, dust marks, etc. while keeping pieces.
    min_area = max(_MIN_CONTOUR_AREA, img_area * 0.0005)
    return [
        c for c in contours
        if min_area <= cv2.contourArea(c) <= _MAX_CONTOUR_AREA_FRACTION * img_area
    ]


def _find_piece_contours_by_non_orange(
    hsv: np.ndarray,
    orange_mask: np.ndarray,
    img_w: int,
    img_h: int,
    img_area: float,
    hsv_low: np.ndarray | None = None,
    hsv_high: np.ndarray | None = None,
    precomputed_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Find piece contours as non-orange blobs within an orange backdrop.

    When the background is a uniform Prusa-orange surface, pieces appear as
    non-orange regions.  This function:
      1. Uses the precomputed cleaned mask from ``_find_orange_backdrop`` when
         available; otherwise rebuilds the mask from HSV via ``cv2.inRange`` and
         morphological cleanup.
      2. Inverts the orange mask to get a ``non_orange_mask``.
      3. Applies morphological cleanup to merge fragmented blobs and remove
         noise from anti-aliased edges.
      4. Returns contours of blobs that are large enough to be a piece but
         not so large that they span the whole image.

    This avoids the intensity-threshold approach used by ``_find_piece_contours``,
    which would trace every bright internal detail on a piece's surface.

    Args:
        hsv:              HSV image (same size as the original).
        orange_mask:      Binary mask where 255 = orange pixel (after morphological
                          cleanup; produced by ``_find_orange_backdrop``).
        img_w:            Image width in pixels.
        img_h:            Image height in pixels.
        img_area:         Total image area in pixels2 (float).
        hsv_low:          Lower HSV bound.  Defaults to ``_ORANGE_HSV_LOW``.
        hsv_high:         Upper HSV bound.  Defaults to ``_ORANGE_HSV_HIGH``.
        precomputed_mask: Pre-cleaned orange mask returned by ``_find_orange_backdrop``
                          (backdrop path).  When provided, the ``cv2.inRange`` +
                          morphological cleanup steps are skipped.

    Returns:
        List of contour arrays, one per detected non-orange blob.
    """
    if hsv_low is None:
        hsv_low = _ORANGE_HSV_LOW
    if hsv_high is None:
        hsv_high = _ORANGE_HSV_HIGH

    if precomputed_mask is not None:
        # Reuse the cleaned mask produced by the backdrop pipeline.
        mask = precomputed_mask
    else:
        # Rebuild the clean orange mask from HSV (same pipeline as _find_orange_backdrop).
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

    # Non-orange pixels are our candidate piece regions.
    non_orange = cv2.bitwise_not(mask)

    # Additional morphological cleanup on the non-orange mask:
    # close gaps that result from orange bleed into piece edges.
    piece_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    non_orange = cv2.morphologyEx(non_orange, cv2.MORPH_CLOSE, piece_kernel)
    non_orange = cv2.morphologyEx(non_orange, cv2.MORPH_OPEN, piece_kernel)

    contours, _ = cv2.findContours(non_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use a size-relative lower bound: at least 0.05% of image area.
    min_area = max(_MIN_CONTOUR_AREA, img_area * 0.0005)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= _MAX_CONTOUR_AREA_FRACTION * img_area):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 20 or bh < 20:
            continue
        # Real puzzle pieces placed on the orange backdrop are completely
        # surrounded by orange and never touch the image border.
        # Border-touching blobs (border strips, shadows at edges) are NOT pieces.
        touches_edge = (
            bx <= 1 or by <= 1
            or bx + bw >= img_w - 1
            or by + bh >= img_h - 1
        )
        if touches_edge:
            continue
        valid.append(c)
    return valid


def segment(
    image: np.ndarray,
    orange_hsv_low: np.ndarray | None = None,
    orange_hsv_high: np.ndarray | None = None,
) -> list[PieceRegion]:
    """Detect surrounding puzzle pieces and the missing center slot.

    Slot / backdrop detection strategy (in order):

    1. **Orange backdrop** - if a Prusa-orange region covers >10% of the image
       the entire background is treated as the orange surface.  Piece contours
       are then found as *non-orange* blobs (``_find_piece_contours_by_non_orange``).
       The backdrop contour defines the overall image boundary used for
       ``slot_bounding_box`` context.

    2. **Orange slot marker** - if a smaller orange region that does NOT touch
       the image border is found, it is treated as a colour-coded slot marker
       placed inside the missing slot.  Piece contours are found via intensity
       thresholding (``_find_piece_contours``).

    3. **Dark-region fallback** - largest interior dark contour (Otsu threshold).
       Piece contours are found via intensity thresholding.

    Args:
        image:           BGR image as numpy array (from cv2.imread).
        orange_hsv_low:  Custom lower HSV bound for orange detection.  When
                         None the module-level ``_ORANGE_HSV_LOW`` is used.
        orange_hsv_high: Custom upper HSV bound for orange detection.  When
                         None the module-level ``_ORANGE_HSV_HIGH`` is used.

    Returns:
        List of PieceRegion objects, one per detected surrounding piece.

    Raises:
        DetectionError: If pieces or missing slot cannot be detected,
                        or if the image is None, empty, or too blurry.
    """
    if image is None or image.size == 0:
        raise DetectionError("Input image is None or empty.")
    if image.ndim < 2:
        raise DetectionError("Input image must be at least 2-dimensional.")

    img_h, img_w = image.shape[:2]
    img_area = float(img_h * img_w)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _check_blur(gray)

    # --- Slot / backdrop detection -------------------------------------------
    slot_contour: np.ndarray | None = None
    use_non_orange_piece_detection = False
    hsv: np.ndarray | None = None
    cleaned_mask: np.ndarray | None = None

    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        orange_contour, is_backdrop, orange_msg, cleaned_mask = _find_orange_backdrop(
            hsv, img_w, img_h,
            gray=gray,
            hsv_low=orange_hsv_low,
            hsv_high=orange_hsv_high,
        )
        print(orange_msg)

        if orange_contour is not None:
            slot_contour = orange_contour
            use_non_orange_piece_detection = is_backdrop

    if slot_contour is None:
        slot_contour = _find_slot_contour(gray, img_w, img_h)
        if slot_contour is not None:
            print("  [slot] Using dark-region fallback.")

    if slot_contour is None:
        raise DetectionError(
            "Could not identify the missing slot or orange backdrop. "
            "Place a piece of Prusa-orange paper in the slot and retake the photo, "
            "or ensure the slot appears as a clearly dark gap with no other dark regions."
        )

    sx, sy, sw, sh = cv2.boundingRect(slot_contour)
    slot_box: tuple[int, int, int, int] = (sx, sy, sw, sh)

    # --- Piece detection -----------------------------------------------------
    if use_non_orange_piece_detection:
        # Use the cleaned mask from _find_orange_backdrop (backdrop path) so the
        # expensive pipeline (fill_holes, morph-close, Gaussian blur) runs only once.
        _low  = orange_hsv_low  if orange_hsv_low  is not None else _ORANGE_HSV_LOW
        _high = orange_hsv_high if orange_hsv_high is not None else _ORANGE_HSV_HIGH
        orange_mask = cv2.inRange(hsv, _low, _high)
        piece_contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, img_w, img_h, img_area,
            hsv_low=orange_hsv_low,
            hsv_high=orange_hsv_high,
            precomputed_mask=cleaned_mask,
        )
    else:
        # Build a filled slot mask to exclude the slot area from piece detection.
        slot_mask_filled = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(slot_mask_filled, [slot_contour], -1, 255, cv2.FILLED)
        piece_contours = _find_piece_contours(gray, img_w, img_h, img_area, slot_mask_filled)

    if not piece_contours:
        raise DetectionError(
            "No surrounding puzzle pieces could be identified. "
            "Ensure the image shows puzzle pieces surrounding a missing slot."
        )

    # --- Build PieceRegion objects -------------------------------------------
    piece_regions: list[PieceRegion] = []
    for piece_id, c in enumerate(piece_contours):
        bx, by, bw, bh = cv2.boundingRect(c)

        epsilon = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        bounding_polygon = approx.reshape(-1, 2)

        margin = 5
        cx1 = max(0, bx - margin)
        cy1 = max(0, by - margin)
        cx2 = min(img_w, bx + bw + margin)
        cy2 = min(img_h, by + bh + margin)
        crop = image[cy1:cy2, cx1:cx2].copy()

        inward_edges = _inward_edges_for_piece((bx, by, bw, bh), slot_box)

        piece_regions.append(
            PieceRegion(
                piece_id=piece_id,
                crop=crop,
                bounding_polygon=bounding_polygon,
                inward_edges=inward_edges,
                slot_bounding_box=slot_box,
            )
        )

    return piece_regions

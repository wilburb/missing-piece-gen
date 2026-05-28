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
# Prusa orange slot-marker colour (OpenCV HSV, H: 0–180, S/V: 0–255)
# Prusa orange ≈ #FA6831 → H 18°, S 80%, V 98%
# The range below is deliberately wide to handle typical indoor lighting variation.
# After each run the detected mean HSV is printed so this range can be refined.
# ---------------------------------------------------------------------------
_ORANGE_HSV_LOW  = np.array([5,  100,  80], dtype=np.uint8)
_ORANGE_HSV_HIGH = np.array([25, 255, 255], dtype=np.uint8)


def _check_blur(gray: np.ndarray) -> None:
    """Raise DetectionError if the image is too blurry."""
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < _BLUR_THRESHOLD:
        raise DetectionError(
            f"Image is too blurry (Laplacian variance {variance:.1f} < {_BLUR_THRESHOLD}). "
            "Please provide a sharper photo."
        )


def _find_slot_contour_by_orange(
    hsv: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray | None, str]:
    """Find the missing-slot contour by looking for the Prusa orange colour marker.

    The caller should place a piece of Prusa-orange paper in the slot before
    photographing; this function detects the most prominent orange region that
    does not touch the image border.

    Returns (contour_or_None, status_message).
    The status message is always populated so the caller can print it.
    """
    mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

    # Morphological cleanup: close small gaps, remove small specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

    orange_px = int(np.count_nonzero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: np.ndarray | None = None
    best_area = 0.0
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
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        return None, (
            f"  [slot] No orange marker found "
            f"(total orange pixels in image: {orange_px}; "
            f"expected HSV {_ORANGE_HSV_LOW.tolist()} – {_ORANGE_HSV_HIGH.tolist()}). "
            "Falling back to dark-region detection."
        )

    # Compute mean HSV of the detected orange region so the user can
    # verify / refine the hardcoded range.
    sx, sy, sw, sh = cv2.boundingRect(best)
    slot_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(slot_mask, [best], -1, 255, cv2.FILLED)
    mean_vals = cv2.mean(hsv, mask=slot_mask)
    mean_h, mean_s, mean_v = mean_vals[0], mean_vals[1], mean_vals[2]

    msg = (
        f"  [slot] Orange marker detected: "
        f"bbox=({sx}, {sy}, {sw}×{sh} px)  area={best_area:.0f} px²\n"
        f"         Detected mean HSV=({mean_h:.0f}, {mean_s:.0f}, {mean_v:.0f})  "
        f"[hardcoded range H {_ORANGE_HSV_LOW[0]}–{_ORANGE_HSV_HIGH[0]}, "
        f"S {_ORANGE_HSV_LOW[1]}–{_ORANGE_HSV_HIGH[1]}, "
        f"V {_ORANGE_HSV_LOW[2]}–{_ORANGE_HSV_HIGH[2]}]"
    )
    return best, msg


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
    """Find contours that correspond to puzzle pieces.

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


def segment(image: np.ndarray) -> list[PieceRegion]:
    """Detect surrounding puzzle pieces and the missing center slot.

    Slot detection strategy (in order):
      1. Prusa orange colour marker — place a piece of Prusa-orange paper in the
         slot before photographing.  Prints the detected mean HSV each run so
         the hardcoded range (_ORANGE_HSV_LOW / _ORANGE_HSV_HIGH) can be refined.
      2. Dark-region fallback — largest interior dark contour (Otsu threshold).

    Args:
        image: BGR image as numpy array (from cv2.imread).

    Returns:
        List of PieceRegion objects, one per detected surrounding piece.

    Raises:
        DetectionError: If pieces or missing slot cannot be detected.
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

    # --- Slot detection: orange marker first, dark-region fallback ---
    slot_contour: np.ndarray | None = None
    slot_mask_filled: np.ndarray | None = None

    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        slot_contour, orange_msg = _find_slot_contour_by_orange(hsv, img_w, img_h)
        print(orange_msg)

    if slot_contour is None:
        slot_contour = _find_slot_contour(gray, img_w, img_h)
        if slot_contour is not None:
            print("  [slot] Using dark-region fallback.")

    if slot_contour is None:
        raise DetectionError(
            "Could not identify the missing slot. "
            "Place a piece of Prusa-orange paper in the slot and retake the photo, "
            "or ensure the slot appears as a clearly dark gap with no other dark regions."
        )

    sx, sy, sw, sh = cv2.boundingRect(slot_contour)
    slot_box: tuple[int, int, int, int] = (sx, sy, sw, sh)

    # Build a filled slot mask to exclude the slot area from piece detection.
    slot_mask_filled = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(slot_mask_filled, [slot_contour], -1, 255, cv2.FILLED)

    # --- Piece detection ---
    piece_contours = _find_piece_contours(gray, img_w, img_h, img_area, slot_mask_filled)
    if not piece_contours:
        raise DetectionError(
            "No surrounding puzzle pieces could be identified. "
            "Ensure the image shows puzzle pieces surrounding the missing slot."
        )

    # --- Build PieceRegion objects ---
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

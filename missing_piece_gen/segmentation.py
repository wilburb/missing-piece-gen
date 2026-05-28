"""Stage 1: Detect and segment surrounding puzzle pieces from a photo."""
import numpy as np
import cv2
from .models import PieceRegion
from .errors import DetectionError

# Laplacian variance below this threshold indicates a blurry image.
_BLUR_THRESHOLD = 50.0

# Minimum contour area (pixels^2) to avoid noise blobs.
_MIN_CONTOUR_AREA = 500

# Maximum contour area fraction of the total image area.
# Contours this large are likely the image border itself.
_MAX_CONTOUR_AREA_FRACTION = 0.70

# How many intensity levels above the estimated background to threshold for pieces.
_PIECE_INTENSITY_OFFSET = 3


def _check_blur(gray: np.ndarray) -> None:
    """Raise DetectionError if the image is too blurry."""
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < _BLUR_THRESHOLD:
        raise DetectionError(
            f"Image is too blurry (Laplacian variance {variance:.1f} < {_BLUR_THRESHOLD}). "
            "Please provide a sharper photo."
        )


def _inward_edges_for_piece(
    piece_box: tuple[int, int, int, int],
    slot_box: tuple[int, int, int, int],
) -> list[str]:
    """Determine which edges of *piece_box* face the missing slot.

    Args:
        piece_box: (x, y, w, h) bounding box of the piece.
        slot_box:  (x, y, w, h) bounding box of the missing slot.

    Returns:
        List of edge direction strings from {"top", "right", "bottom", "left"}.
    """
    px, py, pw, ph = piece_box
    sx, sy, sw, sh = slot_box

    # Centre points
    pc_x = px + pw / 2
    pc_y = py + ph / 2
    sc_x = sx + sw / 2
    sc_y = sy + sh / 2

    dx = sc_x - pc_x  # positive → slot is to the right of piece
    dy = sc_y - pc_y  # positive → slot is below piece

    edges: list[str] = []

    # Include an axis if it contributes at least 40% of the dominant displacement.
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
        # Fallback: assign right as a safe default.
        edges = ["right"]

    return edges


def _find_slot_contour(
    gray: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """Find the contour of the missing slot (largest interior dark region).

    Uses Otsu thresholding to separate the dark slot from the lighter
    background and pieces, then selects the largest dark contour that
    does not touch the image border.

    Returns the contour array or None if no suitable slot is found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The slot is a dark region → it appears as 0 in *binary*.
    # Invert so the slot becomes white for contour detection.
    inv_binary = cv2.bitwise_not(binary)
    dark_contours, _ = cv2.findContours(inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slot_contour = None
    slot_area = 0.0
    for c in dark_contours:
        area = cv2.contourArea(c)
        if area < _MIN_CONTOUR_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Interior: bounding rect must not touch the image border.
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


def _estimate_background_intensity(gray: np.ndarray) -> int:
    """Estimate the dominant background intensity (modal brightness).

    Excludes very dark pixels (< 50) that belong to the slot, so the mode
    reflects the background / piece brightness distribution.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Blank out the dark range to avoid picking up the slot.
    hist[:50] = 0
    return int(np.argmax(hist))


def _find_piece_contours(
    gray: np.ndarray,
    img_w: int,
    img_h: int,
    img_area: float,
) -> list[np.ndarray]:
    """Find contours that correspond to puzzle pieces.

    Strategy: estimate background brightness from the image histogram, then
    threshold at (background + small offset) to isolate pixels that are
    brighter than the background.  Puzzle pieces are typically slightly
    brighter than the surrounding backdrop.

    Returns a list of contour arrays, one per detected piece.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bg_intensity = _estimate_background_intensity(blurred)

    threshold_value = max(0, bg_intensity + _PIECE_INTENSITY_OFFSET)
    _, piece_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [
        c for c in contours
        if _MIN_CONTOUR_AREA <= cv2.contourArea(c) <= _MAX_CONTOUR_AREA_FRACTION * img_area
    ]


def segment(image: np.ndarray) -> list[PieceRegion]:
    """Detect surrounding puzzle pieces and the missing center slot.

    Args:
        image: BGR image as numpy array (from cv2.imread).

    Returns:
        List of PieceRegion objects, one per detected surrounding piece.
        Each PieceRegion includes which edges face the missing slot.

    Raises:
        DetectionError: If pieces or missing slot cannot be detected,
                        or if the image is None, empty, or too blurry.
    """
    # --- 1. Validate input ---------------------------------------------------
    if image is None or image.size == 0:
        raise DetectionError("Input image is None or empty.")

    if image.ndim < 2:
        raise DetectionError("Input image must be at least 2-dimensional.")

    img_h, img_w = image.shape[:2]
    img_area = float(img_h * img_w)

    # --- 2. Convert to grayscale and check sharpness -------------------------
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _check_blur(gray)

    # --- 3. Find the missing slot (interior dark region) --------------------
    slot_contour = _find_slot_contour(gray, img_w, img_h)
    if slot_contour is None:
        raise DetectionError(
            "Could not identify the missing slot. "
            "Ensure the photo has a clearly visible dark gap in the center."
        )

    sx, sy, sw, sh = cv2.boundingRect(slot_contour)
    slot_box: tuple[int, int, int, int] = (sx, sy, sw, sh)

    # --- 4. Find surrounding piece contours ---------------------------------
    piece_contours = _find_piece_contours(gray, img_w, img_h, img_area)
    if not piece_contours:
        raise DetectionError(
            "No surrounding puzzle pieces could be identified. "
            "Ensure the image shows puzzle pieces surrounding a missing slot."
        )

    # --- 5. Build PieceRegion objects ----------------------------------------
    piece_regions: list[PieceRegion] = []

    for piece_id, c in enumerate(piece_contours):
        bx, by, bw, bh = cv2.boundingRect(c)

        # Approximate the contour as a polygon.
        epsilon = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        bounding_polygon = approx.reshape(-1, 2)

        # Crop the original image to the piece bounding box (with small margin).
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

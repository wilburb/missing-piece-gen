"""Tests for missing_piece_gen.segmentation (Stage 1)."""
import numpy as np
import cv2
import pytest

from missing_piece_gen.segmentation import (
    segment,
    calibrate_orange_hsv,
    calibrate_from_ruler,
    _find_orange_backdrop,
    _find_piece_contours_by_non_orange,
    _ORANGE_HSV_LOW,
    _ORANGE_HSV_HIGH,
)
from missing_piece_gen.models import PieceRegion
from missing_piece_gen.errors import DetectionError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_test_image() -> np.ndarray:
    """Create a synthetic puzzle-like image: white pieces around a dark center slot.

    Layout (400x400 px):
      - Mid-gray background
      - Dark rectangle in the center simulates the missing slot
      - Two lighter rectangles (left and right of the slot) simulate surrounding pieces
    """
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200  # gray background
    # Draw dark slot in center
    cv2.rectangle(img, (150, 150), (250, 250), (30, 30, 30), -1)
    # Draw lighter pieces around it
    cv2.rectangle(img, (50, 50), (140, 390), (220, 220, 220), -1)   # left piece
    cv2.rectangle(img, (260, 50), (350, 390), (220, 220, 220), -1)  # right piece
    return img


def make_blurry_image() -> np.ndarray:
    """Return a heavily blurred image with very low Laplacian variance."""
    base = make_test_image()
    # Apply large kernel blur many times to make Laplacian variance drop well below 50
    blurred = base.copy()
    for _ in range(20):
        blurred = cv2.GaussianBlur(blurred, (31, 31), 0)
    return blurred


# ---------------------------------------------------------------------------
# Tests: valid image
# ---------------------------------------------------------------------------


def test_segment_returns_list_of_piece_regions():
    """Given a valid synthetic image, segment() returns a non-empty list."""
    img = make_test_image()
    result = segment(img)
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert isinstance(item, PieceRegion)


def test_piece_regions_have_required_fields():
    """Each PieceRegion has the required fields with correct types."""
    img = make_test_image()
    regions = segment(img)
    for region in regions:
        # piece_id is an int
        assert isinstance(region.piece_id, int)
        # crop is a numpy array with 3 channels
        assert isinstance(region.crop, np.ndarray)
        assert region.crop.ndim == 3
        assert region.crop.shape[2] == 3
        # bounding_polygon is a numpy array with shape (N, 2)
        assert isinstance(region.bounding_polygon, np.ndarray)
        assert region.bounding_polygon.ndim == 2
        assert region.bounding_polygon.shape[1] == 2
        # inward_edges is a non-empty list of strings
        assert isinstance(region.inward_edges, list)
        assert len(region.inward_edges) > 0
        for edge in region.inward_edges:
            assert edge in {"top", "right", "bottom", "left"}
        # slot_bounding_box is a 4-tuple or None
        if region.slot_bounding_box is not None:
            assert isinstance(region.slot_bounding_box, tuple)
            assert len(region.slot_bounding_box) == 4


def test_piece_ids_are_unique():
    """piece_id values across all returned PieceRegions are unique."""
    img = make_test_image()
    regions = segment(img)
    ids = [r.piece_id for r in regions]
    assert len(ids) == len(set(ids))


def test_slot_bounding_box_populated():
    """slot_bounding_box is populated and plausible."""
    img = make_test_image()
    regions = segment(img)
    for region in regions:
        assert region.slot_bounding_box is not None
        x, y, w, h = region.slot_bounding_box
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


def test_segment_raises_on_none_input():
    """segment(None) raises DetectionError."""
    with pytest.raises(DetectionError):
        segment(None)


def test_segment_raises_on_empty_array():
    """segment() raises DetectionError for an empty numpy array."""
    with pytest.raises(DetectionError):
        segment(np.array([]))


def test_segment_raises_on_blurry_image():
    """segment() raises DetectionError when the image is too blurry."""
    blurry = make_blurry_image()
    with pytest.raises(DetectionError, match="blurry"):
        segment(blurry)


def test_segment_raises_on_uniform_image():
    """segment() raises DetectionError for a completely uniform (featureless) image."""
    uniform = np.full((300, 300, 3), 128, dtype=np.uint8)
    with pytest.raises(DetectionError):
        segment(uniform)


# ---------------------------------------------------------------------------
# Helpers for orange-backdrop tests
# ---------------------------------------------------------------------------


def _orange_bgr() -> tuple[int, int, int]:
    """Return a BGR colour that falls within the Prusa-orange HSV range."""
    # Prusa orange #FA6831 → BGR (49, 104, 250)
    return (49, 104, 250)


def make_orange_backdrop_image(
    width: int = 400,
    height: int = 400,
    piece_rects: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Create a synthetic image with a full Prusa-orange backdrop.

    The orange region fills the entire canvas (touches all four borders).
    Non-orange rectangles represent puzzle pieces placed on the surface.

    Args:
        width:       Image width in pixels.
        height:      Image height in pixels.
        piece_rects: List of (x1, y1, x2, y2) rectangles to draw as
                     dark-gray pieces on top of the orange backdrop.
                     Defaults to two rectangles left/right of centre.

    Returns:
        BGR image as a numpy array.
    """
    bgr = _orange_bgr()
    img = np.full((height, width, 3), bgr, dtype=np.uint8)

    if piece_rects is None:
        cx, cy = width // 2, height // 2
        piece_rects = [
            (20, 20, cx - 20, height - 20),    # left piece
            (cx + 20, 20, width - 20, height - 20),  # right piece
        ]

    piece_color = (60, 60, 60)  # dark gray — clearly non-orange
    for x1, y1, x2, y2 in piece_rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), piece_color, -1)

    return img


# ---------------------------------------------------------------------------
# Tests: orange backdrop detection (issue #30)
# ---------------------------------------------------------------------------


def test_orange_backdrop_detected_when_touching_edges():
    """_find_orange_backdrop returns is_backdrop=True even when orange spans the image.

    Regression test for issue #30: the old touches_edge guard would reject a
    full-background orange region because its bounding rect necessarily touches
    all four image borders.  The new implementation accepts it when the orange
    covers > _ORANGE_BACKDROP_FRACTION of the image area.
    """
    img = make_orange_backdrop_image(piece_rects=[(20, 20, 180, 380), (220, 20, 380, 380)])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    contour, is_backdrop, msg = _find_orange_backdrop(hsv, w, h)

    assert contour is not None, f"Expected orange contour to be found; msg={msg!r}"
    assert is_backdrop is True, (
        f"Expected is_backdrop=True for a full-canvas orange region; msg={msg!r}"
    )


def test_orange_backdrop_contour_covers_most_of_image():
    """The detected backdrop contour should span most of the image."""
    img = make_orange_backdrop_image()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    img_area = float(h * w)

    contour, is_backdrop, _ = _find_orange_backdrop(hsv, w, h)

    assert contour is not None
    assert is_backdrop is True
    # The orange fills the entire canvas so its contour area should be large.
    area = cv2.contourArea(contour)
    assert area / img_area >= 0.10, f"Expected contour area >= 10% of image, got {area / img_area:.3f}"


def test_segment_orange_backdrop_finds_piece_contours():
    """segment() detects pieces as non-orange blobs when the backdrop is orange.

    Verifies end-to-end that:
    - Two piece-shaped rectangles on an orange surface are returned as PieceRegions.
    - slot_bounding_box is populated.
    - piece_id values are unique.
    """
    img = make_orange_backdrop_image(
        width=400,
        height=400,
        piece_rects=[(20, 20, 180, 380), (220, 20, 380, 380)],
    )
    regions = segment(img)

    assert len(regions) >= 1, "Expected at least one piece region on an orange backdrop"
    for region in regions:
        assert isinstance(region, PieceRegion)
        assert region.slot_bounding_box is not None
        x, y, w, h = region.slot_bounding_box
        assert w > 0 and h > 0

    ids = [r.piece_id for r in regions]
    assert len(ids) == len(set(ids)), "piece_id values should be unique"


def test_segment_orange_backdrop_piece_crop_is_not_orange():
    """Piece crops from an orange-backdrop image should NOT be predominantly orange.

    This guards against the regression where the orange backdrop itself was
    returned as a 'piece' rather than the actual non-orange objects on it.
    """
    img = make_orange_backdrop_image(
        width=400,
        height=400,
        piece_rects=[(20, 20, 180, 380), (220, 20, 380, 380)],
    )
    regions = segment(img)

    for region in regions:
        crop_hsv = cv2.cvtColor(region.crop, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(crop_hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)
        orange_fraction = np.count_nonzero(orange_mask) / float(orange_mask.size)
        assert orange_fraction < 0.5, (
            f"Piece crop (id={region.piece_id}) is >50% orange "
            f"({orange_fraction:.2%}); the backdrop was likely detected as a piece."
        )


# ---------------------------------------------------------------------------
# Tests: thin sliver filter in _find_piece_contours_by_non_orange (issue #32)
# ---------------------------------------------------------------------------


def _make_orange_hsv_image(width: int, height: int) -> np.ndarray:
    """Return an HSV image filled entirely with Prusa orange."""
    bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


class TestThinSliverFilter:
    """Regression tests for issue #32: thin border slivers must be rejected."""

    def test_thin_horizontal_sliver_is_filtered_out(self):
        """A sliver blob < 20px tall must be rejected even if its area passes the filter.

        Simulates the 3px tall × 1500px wide non-orange strip that used to
        pass the area check in _find_piece_contours_by_non_orange, leading to a
        degenerate crop and an 'Unknown C++ exception' from cv2.GaussianBlur.
        """
        width, height = 400, 400
        img_area = float(width * height)

        # Build an image that is entirely Prusa orange except for a very thin
        # horizontal strip near the top (3 px tall × full width = 1200 px² area,
        # which exceeds _MIN_CONTOUR_AREA=2000 when scaled, but we use a wider
        # strip to guarantee it passes area while still being < 20px tall).
        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # Insert a 10px tall × 400px wide non-orange strip at the top edge.
        # Area = 10 * 400 = 4000 px² — above _MIN_CONTOUR_AREA=2000.
        bgr[0:10, :] = (60, 60, 60)  # dark gray (non-orange)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        # The 10px-tall sliver must be filtered out (bh < 20px).
        for c in contours:
            _, _, bw, bh = cv2.boundingRect(c)
            assert bw >= 20 and bh >= 20, (
                f"Thin sliver with bounding rect {bw}×{bh} should have been filtered out"
            )

    def test_thin_vertical_sliver_is_filtered_out(self):
        """A sliver blob < 20px wide must be rejected (issue #32)."""
        width, height = 400, 400
        img_area = float(width * height)

        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # Insert a 10px wide × 400px tall non-orange strip on the left edge.
        bgr[:, 0:10] = (60, 60, 60)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        for c in contours:
            _, _, bw, bh = cv2.boundingRect(c)
            assert bw >= 20 and bh >= 20, (
                f"Thin sliver with bounding rect {bw}×{bh} should have been filtered out"
            )

    def test_valid_piece_blob_is_not_filtered_out(self):
        """A blob >= 20px in both dimensions must NOT be filtered out."""
        width, height = 400, 400
        img_area = float(width * height)

        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # Insert a 100×100 non-orange piece well inside the frame.
        bgr[50:150, 50:150] = (60, 60, 60)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        assert len(contours) >= 1, (
            "A 100×100 non-orange blob should not be filtered out by the dimension check"
        )


# ---------------------------------------------------------------------------
# Tests: border-touching blob filter in _find_piece_contours_by_non_orange
# (issue #34)
# ---------------------------------------------------------------------------


class TestBorderTouchingBlobFilter:
    """Regression tests for issue #34: blobs touching the image border must be rejected."""

    def test_border_touching_blob_is_filtered_out(self):
        """A non-orange blob that touches the image border must be rejected.

        Real puzzle pieces on an orange backdrop are completely surrounded by
        orange and never touch the image border.  Border-touching blobs (border
        strips, shadows at edges) are noise and must be excluded.
        """
        width, height = 400, 400
        img_area = float(width * height)

        # Orange backdrop with a large non-orange blob that starts at x=0 (touches left edge).
        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # 100×200 dark blob anchored at the left border (bx=0).
        bgr[100:300, 0:100] = (60, 60, 60)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        # The border-touching blob must be rejected.
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            assert bx > 1 and by > 1, (
                f"Border-touching blob at bx={bx}, by={by} should have been filtered out"
            )
            assert bx + bw < width - 1 and by + bh < height - 1, (
                f"Border-touching blob reaching bx+bw={bx+bw}, by+bh={by+bh} "
                f"should have been filtered out (img {width}×{height})"
            )

    def test_border_touching_top_blob_is_filtered_out(self):
        """A non-orange blob touching the top border is rejected (issue #34)."""
        width, height = 400, 400
        img_area = float(width * height)

        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # 400×50 dark strip at the very top (by=0).
        bgr[0:50, :] = (60, 60, 60)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            assert by > 1, (
                f"Border-touching blob at by={by} should have been filtered out"
            )

    def test_interior_blob_is_not_filtered_out(self):
        """A non-orange blob fully interior (not touching any edge) must NOT be rejected."""
        width, height = 400, 400
        img_area = float(width * height)

        bgr = np.full((height, width, 3), (49, 104, 250), dtype=np.uint8)
        # 100×100 dark blob fully inside the frame (at least 5px from every edge).
        bgr[50:150, 50:150] = (60, 60, 60)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, _ORANGE_HSV_LOW, _ORANGE_HSV_HIGH)

        contours = _find_piece_contours_by_non_orange(
            hsv, orange_mask, width, height, img_area
        )

        assert len(contours) >= 1, (
            "A 100×100 interior non-orange blob must not be filtered by the border-touch guard"
        )


# ---------------------------------------------------------------------------
# Tests: calibrate_orange_hsv and --orange-ref integration (issue #36)
# ---------------------------------------------------------------------------


def test_calibrate_orange_hsv_basic():
    """calibrate_orange_hsv returns valid uint8 arrays with low[i] <= high[i].

    Creates a solid Prusa-orange BGR image and verifies that the returned
    (low, high) pair satisfies the contract: all values are within the valid
    OpenCV HSV range and the lower bound never exceeds the upper bound.
    """
    # Prusa orange BGR: (49, 104, 250)
    orange_bgr = np.full((100, 100, 3), (49, 104, 250), dtype=np.uint8)
    low, high = calibrate_orange_hsv(orange_bgr)

    assert low.dtype == np.uint8, "low should be uint8"
    assert high.dtype == np.uint8, "high should be uint8"
    assert low.shape == (3,), "low should be a 1-D array of length 3"
    assert high.shape == (3,), "high should be a 1-D array of length 3"

    # H channel: 0–180
    assert 0 <= int(low[0]) <= 180, f"low H={low[0]} out of range"
    assert 0 <= int(high[0]) <= 180, f"high H={high[0]} out of range"
    # S and V channels: 0–255
    for i in (1, 2):
        assert 0 <= int(low[i]) <= 255, f"low[{i}]={low[i]} out of range"
        assert 0 <= int(high[i]) <= 255, f"high[{i}]={high[i]} out of range"

    # low[i] must not exceed high[i] for any channel
    for i in range(3):
        assert int(low[i]) <= int(high[i]), (
            f"low[{i}]={low[i]} > high[{i}]={high[i]}"
        )


def test_calibrate_orange_hsv_overrides_detection():
    """Calibrated bounds from a reddish-orange reference allow _find_orange_backdrop
    to detect a non-standard orange that the built-in range would miss.

    Uses a reddish-orange with H≈3 (well below the built-in lower bound of H=5)
    as the backdrop and verifies that _find_orange_backdrop finds it when supplied
    with the calibrated bounds.
    """
    # Reddish-orange: H≈3 in OpenCV (6° in degrees).
    # BGR equivalent for HSV (3, 220, 230): approximately (38, 38, 230) in BGR.
    # Use cv2 to construct a precise HSV→BGR conversion.
    hsv_ref = np.full((100, 100, 3), (3, 220, 230), dtype=np.uint8)
    bgr_ref = cv2.cvtColor(hsv_ref, cv2.COLOR_HSV2BGR)

    # The built-in range (H 5–25) may miss H=3.  Calibrate from the reference.
    low, high = calibrate_orange_hsv(bgr_ref)

    # Build a 400×400 backdrop image using the same reddish-orange.
    img = np.full((400, 400, 3), bgr_ref[0, 0].tolist(), dtype=np.uint8)
    # Place two dark pieces on it so it qualifies as a backdrop (> 10% orange).
    img[50:150, 50:150] = (60, 60, 60)
    img[200:300, 200:300] = (60, 60, 60)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contour, is_backdrop, msg = _find_orange_backdrop(hsv, 400, 400, hsv_low=low, hsv_high=high)

    assert contour is not None, (
        f"Expected a contour with calibrated bounds; msg={msg!r}"
    )
    assert is_backdrop is True, (
        f"Expected is_backdrop=True for a full reddish-orange canvas; msg={msg!r}"
    )


def test_segment_uses_custom_hsv_bounds():
    """segment() passes custom HSV bounds through to orange detection.

    Constructs a reddish-orange backdrop image and verifies that segment()
    succeeds when supplied with calibrated bounds derived from a reference image
    of the same colour.  Without calibration the pipeline would fall back to
    dark-region detection (or raise DetectionError).
    """
    # Same reddish-orange as above (H≈3).
    hsv_ref = np.full((100, 100, 3), (3, 220, 230), dtype=np.uint8)
    bgr_ref = cv2.cvtColor(hsv_ref, cv2.COLOR_HSV2BGR)
    ref_color = bgr_ref[0, 0].tolist()

    low, high = calibrate_orange_hsv(bgr_ref)

    # Build a 400×400 backdrop with two large interior non-orange pieces.
    img = np.full((400, 400, 3), ref_color, dtype=np.uint8)
    img[30:180, 30:180] = (60, 60, 60)   # left piece (well inside the frame)
    img[30:180, 220:370] = (60, 60, 60)  # right piece (well inside the frame)

    # Add a fine noise pattern to raise the Laplacian variance above the blur
    # threshold so _check_blur does not reject this synthetic image.
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 11, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    regions = segment(img, orange_hsv_low=low, orange_hsv_high=high)

    assert len(regions) >= 1, (
        "segment() with calibrated bounds should detect pieces on a reddish-orange backdrop"
    )
    for region in regions:
        assert isinstance(region, PieceRegion)
        assert region.slot_bounding_box is not None


# ---------------------------------------------------------------------------
# Tests: calibrate_from_ruler (issue #39)
# ---------------------------------------------------------------------------


def test_calibrate_from_ruler_detects_peaks():
    """calibrate_from_ruler returns px/mm close to the known tick spacing.

    Creates a synthetic grayscale image with a bright vertical stripe at
    x=50–110 and regularly-spaced bright horizontal lines every 15px.
    The function should detect the 15px gaps and return a value close to 15.0.
    """
    tick_spacing = 15
    height = 400
    width = 200
    # Dark background
    img = np.zeros((height, width), dtype=np.uint8)
    # Bright stripe in the ruler column range
    img[:, 50:110] = 50
    # Add bright horizontal tick marks every tick_spacing pixels
    for y in range(0, height, tick_spacing):
        img[y, 50:110] = 200

    result = calibrate_from_ruler(img)

    assert result is not None, "Expected a float result for a well-spaced ruler image"
    assert abs(result - tick_spacing) < 2.0, (
        f"Expected px/mm close to {tick_spacing}, got {result:.2f}"
    )


def test_calibrate_from_ruler_too_few_peaks_returns_none():
    """calibrate_from_ruler returns None when fewer than 3 usable gaps are found.

    Creates a profile with only 2 bright lines (yielding 1 gap), which is
    below the minimum of 3 gaps required to return a calibration value.
    """
    height = 200
    width = 200
    img = np.zeros((height, width), dtype=np.uint8)
    img[:, 50:110] = 50
    # Only 2 tick marks → 1 gap → fewer than 3 usable gaps
    img[20, 50:110] = 200
    img[35, 50:110] = 200

    result = calibrate_from_ruler(img)

    assert result is None, (
        f"Expected None for profile with only 2 peaks, got {result!r}"
    )


def test_calibrate_from_ruler_image_too_narrow():
    """calibrate_from_ruler returns None when the image is narrower than 111px."""
    # Image with width < 111 — cannot access x=50:110
    img = np.zeros((300, 100), dtype=np.uint8)

    result = calibrate_from_ruler(img)

    assert result is None, (
        f"Expected None for image width < 111, got {result!r}"
    )

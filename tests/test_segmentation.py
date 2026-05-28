"""Tests for missing_piece_gen.segmentation (Stage 1)."""
import numpy as np
import cv2
import pytest

from missing_piece_gen.segmentation import segment
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

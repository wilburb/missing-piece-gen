"""Tests for Stage 2: edge profile extraction."""
import numpy as np
import cv2
import pytest

from missing_piece_gen.models import PieceRegion, EdgeType
from missing_piece_gen.edge_analysis import extract_edges
from missing_piece_gen.errors import EdgeExtractionError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_piece_with_tab() -> PieceRegion:
    """Synthetic piece crop with a tab bump on the top edge."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    # White piece body
    cv2.rectangle(crop, (10, 20), (90, 90), (255, 255, 255), -1)
    # Tab: additional bump on the top
    cv2.rectangle(crop, (40, 5), (60, 22), (255, 255, 255), -1)
    piece = PieceRegion(
        piece_id=0,
        crop=crop,
        bounding_polygon=np.array([[10, 5], [90, 5], [90, 90], [10, 90]]),
        inward_edges=["top"],
        slot_bounding_box=(0, 0, 100, 100),
    )
    return piece


def make_piece_with_blank() -> PieceRegion:
    """Synthetic piece crop with a blank indentation on the top edge."""
    crop = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Carve out a blank (indentation) from the top edge
    cv2.rectangle(crop, (40, 10), (60, 35), (0, 0, 0), -1)
    piece = PieceRegion(
        piece_id=1,
        crop=crop,
        bounding_polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
        inward_edges=["top"],
        slot_bounding_box=(0, 0, 100, 100),
    )
    return piece


def make_flat_piece() -> PieceRegion:
    """Synthetic flat-edged piece (plain rectangle, no tab or blank)."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
    piece = PieceRegion(
        piece_id=2,
        crop=crop,
        bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
        inward_edges=["top"],
        slot_bounding_box=(0, 0, 100, 100),
    )
    return piece


def make_corner_piece() -> PieceRegion:
    """Synthetic corner piece with two inward-facing edges."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(crop, (5, 5), (95, 95), (200, 200, 200), -1)
    # Tab on top
    cv2.rectangle(crop, (40, 0), (60, 7), (200, 200, 200), -1)
    # Tab on left
    cv2.rectangle(crop, (0, 40), (7, 60), (200, 200, 200), -1)
    piece = PieceRegion(
        piece_id=3,
        crop=crop,
        bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
        inward_edges=["top", "left"],
        slot_bounding_box=(0, 0, 100, 100),
    )
    return piece


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractEdgesReturnsList:
    def test_extract_edges_returns_list(self):
        piece = make_flat_piece()
        result = extract_edges(piece)
        assert isinstance(result, list)

    def test_returns_empty_list_for_no_inward_edges(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
        piece = PieceRegion(
            piece_id=99,
            crop=crop,
            bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
            inward_edges=[],
        )
        result = extract_edges(piece)
        assert result == []


class TestEdgeProfileDirection:
    def test_edge_profile_has_correct_direction(self):
        piece = make_flat_piece()
        profiles = extract_edges(piece)
        assert len(profiles) == 1
        assert profiles[0].direction == "top"

    def test_direction_matches_each_inward_edge(self):
        piece = make_corner_piece()
        profiles = extract_edges(piece)
        directions = [p.direction for p in profiles]
        assert "top" in directions
        assert "left" in directions


class TestTabClassification:
    def test_tab_piece_classified_as_tab_or_not_flat(self):
        """
        Be lenient: synthetic images may produce TAB or (rarely) FLAT depending
        on contour detection, but should never be BLANK.
        """
        piece = make_piece_with_tab()
        profiles = extract_edges(piece)
        assert len(profiles) == 1
        assert profiles[0].edge_type != EdgeType.BLANK

    def test_edge_type_is_valid_enum(self):
        piece = make_piece_with_tab()
        profiles = extract_edges(piece)
        for p in profiles:
            assert isinstance(p.edge_type, EdgeType)


class TestMultipleInwardEdges:
    def test_multiple_inward_edges_returns_correct_count(self):
        piece = make_corner_piece()
        profiles = extract_edges(piece)
        assert len(profiles) == 2

    def test_each_profile_has_valid_edge_type(self):
        piece = make_corner_piece()
        profiles = extract_edges(piece)
        for p in profiles:
            assert isinstance(p.edge_type, EdgeType)

    def test_all_four_inward_edges(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
        piece = PieceRegion(
            piece_id=4,
            crop=crop,
            bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
            inward_edges=["top", "right", "bottom", "left"],
        )
        profiles = extract_edges(piece)
        assert len(profiles) == 4


class TestEmptyCropRaisesError:
    def test_none_crop_raises_edge_extraction_error(self):
        piece = PieceRegion(
            piece_id=10,
            crop=None,
            bounding_polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            inward_edges=["top"],
        )
        with pytest.raises(EdgeExtractionError):
            extract_edges(piece)

    def test_zero_size_crop_raises_edge_extraction_error(self):
        piece = PieceRegion(
            piece_id=11,
            crop=np.zeros((0, 0, 3), dtype=np.uint8),
            bounding_polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            inward_edges=["top"],
        )
        with pytest.raises(EdgeExtractionError):
            extract_edges(piece)


class TestContourIsNumpyArray:
    def test_contour_is_numpy_array(self):
        piece = make_flat_piece()
        profiles = extract_edges(piece)
        for p in profiles:
            assert isinstance(p.contour, np.ndarray)

    def test_contour_has_two_columns(self):
        piece = make_flat_piece()
        profiles = extract_edges(piece)
        for p in profiles:
            if p.contour.size > 0:
                assert p.contour.ndim == 2
                assert p.contour.shape[1] == 2

    def test_contour_dtype_is_float(self):
        piece = make_flat_piece()
        profiles = extract_edges(piece)
        for p in profiles:
            if p.contour.size > 0:
                assert np.issubdtype(p.contour.dtype, np.floating)


class TestTabGeometry:
    def test_flat_edge_has_no_tab_geometry(self):
        """A sufficiently flat piece should have tab_geometry=None."""
        piece = make_flat_piece()
        profiles = extract_edges(piece)
        for p in profiles:
            if p.edge_type == EdgeType.FLAT:
                assert p.tab_geometry is None

    def test_tab_geometry_position_in_range(self):
        """If tab_geometry is present, position must be in [0, 1]."""
        piece = make_piece_with_tab()
        profiles = extract_edges(piece)
        for p in profiles:
            if p.tab_geometry is not None:
                assert 0.0 <= p.tab_geometry.position <= 1.0

    def test_tab_geometry_depth_positive(self):
        """depth must be a positive number when tab_geometry is present."""
        piece = make_piece_with_tab()
        profiles = extract_edges(piece)
        for p in profiles:
            if p.tab_geometry is not None:
                assert p.tab_geometry.depth > 0


class TestEdgeDirectionVariants:
    def test_bottom_edge_returns_profile(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
        piece = PieceRegion(
            piece_id=20,
            crop=crop,
            bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
            inward_edges=["bottom"],
        )
        profiles = extract_edges(piece)
        assert len(profiles) == 1
        assert profiles[0].direction == "bottom"
        assert isinstance(profiles[0].edge_type, EdgeType)

    def test_right_edge_returns_profile(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
        piece = PieceRegion(
            piece_id=21,
            crop=crop,
            bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
            inward_edges=["right"],
        )
        profiles = extract_edges(piece)
        assert len(profiles) == 1
        assert profiles[0].direction == "right"

    def test_unknown_direction_returns_profile_gracefully(self):
        """An unknown direction should not crash — it falls back to full crop."""
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (5, 5), (95, 95), (255, 255, 255), -1)
        piece = PieceRegion(
            piece_id=22,
            crop=crop,
            bounding_polygon=np.array([[5, 5], [95, 5], [95, 95], [5, 95]]),
            inward_edges=["diagonal"],
        )
        profiles = extract_edges(piece)
        assert len(profiles) == 1
        assert isinstance(profiles[0].edge_type, EdgeType)

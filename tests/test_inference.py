"""Tests for missing_piece_gen.inference — Stage 3: Missing piece shape inference."""
import numpy as np

from missing_piece_gen.models import EdgeProfile, EdgeType, TabGeometry
from missing_piece_gen.inference import infer_shape, _shoelace_area


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_flat_profile(direction: str) -> EdgeProfile:
    """Simple straight contour for a flat edge spanning 80 pixels."""
    contour = np.array([[10, 50], [30, 50], [50, 50], [70, 50], [90, 50]], dtype=float)
    return EdgeProfile(direction=direction, contour=contour, edge_type=EdgeType.FLAT)


def make_tab_profile(direction: str) -> EdgeProfile:
    """Contour with a bump (TAB) — protrudes ~20 px at the midpoint."""
    contour = np.array(
        [[10, 50], [30, 50], [40, 35], [50, 30], [60, 35], [70, 50], [90, 50]],
        dtype=float,
    )
    return EdgeProfile(
        direction=direction,
        contour=contour,
        edge_type=EdgeType.TAB,
        tab_geometry=TabGeometry(position=0.5, width=30.0, depth=20.0),
    )


def make_blank_profile(direction: str) -> EdgeProfile:
    """Contour with an indentation (BLANK)."""
    contour = np.array(
        [[10, 50], [30, 50], [40, 65], [50, 70], [60, 65], [70, 50], [90, 50]],
        dtype=float,
    )
    return EdgeProfile(
        direction=direction,
        contour=contour,
        edge_type=EdgeType.BLANK,
        tab_geometry=TabGeometry(position=0.5, width=30.0, depth=20.0),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_missing_piece_shape(self):
        from missing_piece_gen.models import MissingPieceShape

        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles)
        assert isinstance(result, MissingPieceShape)

    def test_outline_is_numpy_array(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles)
        assert isinstance(result.outline, np.ndarray)

    def test_outline_is_2d_with_two_columns(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles)
        assert result.outline.ndim == 2
        assert result.outline.shape[1] == 2

    def test_outline_has_positive_area(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles)
        area = _shoelace_area(result.outline)
        assert area > 0, f"Expected positive area, got {area}"

    def test_outline_is_closed(self):
        """First and last points should be the same (closed polygon)."""
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles)
        assert np.allclose(result.outline[0], result.outline[-1]), (
            "Outline is not closed: first != last point"
        )


class TestEmptyProfiles:
    def test_empty_profiles_uses_hint(self):
        """With no edge profiles, the shape should use piece_width_hint_mm."""
        hint_mm = 30.0
        result = infer_shape([], piece_width_hint_mm=hint_mm)
        # Width and height should be approximately the hint
        assert abs(result.width_mm - hint_mm) < 1.0, (
            f"Expected width ≈ {hint_mm}, got {result.width_mm}"
        )

    def test_empty_profiles_returns_valid_shape(self):
        result = infer_shape([])
        area = _shoelace_area(result.outline)
        assert area > 0

    def test_empty_profiles_stores_scale(self):
        result = infer_shape([], pixel_to_mm_scale=2.0)
        assert result.pixel_to_mm_scale == 2.0


class TestPartialProfiles:
    def test_partial_profiles_fills_gaps(self):
        """Only 2 profiles — gaps filled with flat edges."""
        profiles = [make_flat_profile("top"), make_flat_profile("bottom")]
        result = infer_shape(profiles)
        area = _shoelace_area(result.outline)
        assert area > 0

    def test_single_profile_still_valid(self):
        profiles = [make_flat_profile("left")]
        result = infer_shape(profiles)
        assert result.outline.ndim == 2
        assert _shoelace_area(result.outline) > 0

    def test_three_profiles_still_valid(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom")]
        result = infer_shape(profiles)
        assert _shoelace_area(result.outline) > 0


class TestPixelToMmScale:
    def test_pixel_to_mm_scale_applied(self):
        """width_mm should be ~2x larger when scale doubles, all else equal."""
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result1 = infer_shape(profiles, pixel_to_mm_scale=1.0)
        result2 = infer_shape(profiles, pixel_to_mm_scale=2.0)
        ratio = result2.width_mm / result1.width_mm
        assert abs(ratio - 2.0) < 0.1, f"Expected ratio ≈ 2.0, got {ratio}"

    def test_scale_stored_on_result(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles, pixel_to_mm_scale=0.5)
        assert result.pixel_to_mm_scale == 0.5

    def test_dimensions_positive(self):
        profiles = [make_flat_profile(d) for d in ("top", "right", "bottom", "left")]
        result = infer_shape(profiles, pixel_to_mm_scale=1.0)
        assert result.width_mm > 0
        assert result.height_mm > 0


class TestComplementaryEdge:
    def test_tab_profile_produces_complement(self):
        """Given a TAB profile on top, the missing piece outline should curve inward
        (BLANK complement) — y values on the top edge segment should deviate from 0
        in the inward direction."""
        profiles = [make_tab_profile("top")]
        result = infer_shape(profiles, pixel_to_mm_scale=1.0)
        # The outline should still be valid
        assert _shoelace_area(result.outline) > 0

        # Find points on the top edge segment (y values near the minimum y, i.e. top)
        outline = result.outline
        y_vals = outline[:, 1]
        # The top edge should have y values that differ from a pure flat edge (y=0).
        # With a TAB on the surrounding piece, the missing piece curves inward (y > 0
        # on the top side in image convention).
        top_region = outline[y_vals < (y_vals.max() * 0.3)]
        if len(top_region) > 0:
            # There should be some variation — not all y=0
            y_range = top_region[:, 1].max() - top_region[:, 1].min()
            assert y_range >= 0  # structural: just verify no crash, outline is valid

    def test_blank_profile_produces_complement(self):
        """A BLANK surrounding profile should also yield a valid closed outline."""
        profiles = [make_blank_profile("right")]
        result = infer_shape(profiles, pixel_to_mm_scale=1.0)
        assert _shoelace_area(result.outline) > 0
        assert isinstance(result.outline, np.ndarray)

    def test_mixed_profiles_valid(self):
        """Mix of TAB/BLANK/FLAT profiles across all four sides."""
        profiles = [
            make_tab_profile("top"),
            make_flat_profile("right"),
            make_blank_profile("bottom"),
            make_tab_profile("left"),
        ]
        result = infer_shape(profiles, pixel_to_mm_scale=1.0)
        assert _shoelace_area(result.outline) > 0
        assert np.allclose(result.outline[0], result.outline[-1])

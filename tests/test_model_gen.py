"""Tests for Stage 4: model_gen.py — 3D extrusion and export."""
import numpy as np
import pytest
import tempfile
import trimesh
from pathlib import Path

from missing_piece_gen.models import MissingPieceShape
from missing_piece_gen.model_gen import generate
from missing_piece_gen.errors import ModelGenerationError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_square_shape(size_mm: float = 20.0) -> MissingPieceShape:
    """Simple square MissingPieceShape for testing."""
    outline = np.array([
        [0.0, 0.0],
        [size_mm, 0.0],
        [size_mm, size_mm],
        [0.0, size_mm],
        [0.0, 0.0],
    ])
    return MissingPieceShape(
        outline=outline,
        width_mm=size_mm,
        height_mm=size_mm,
        pixel_to_mm_scale=1.0,
    )


def make_irregular_shape() -> MissingPieceShape:
    """Irregular polygon to simulate a real puzzle piece outline."""
    outline = np.array([
        [0, 0], [5, 0], [8, 2], [10, 0], [15, 0],
        [15, 10], [10, 12], [8, 10], [5, 12], [0, 10], [0, 0],
    ], dtype=float)
    return MissingPieceShape(
        outline=outline,
        width_mm=15.0,
        height_mm=12.0,
        pixel_to_mm_scale=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_stl_creates_file():
    """STL output file is created and non-empty."""
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "piece.stl"
        generate(shape, out, format="stl")
        assert out.exists(), "STL file was not created"
        assert out.stat().st_size > 0, "STL file is empty"


def test_generate_obj_creates_file():
    """OBJ output file is created and non-empty."""
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "piece.obj"
        generate(shape, out, format="obj")
        assert out.exists(), "OBJ file was not created"
        assert out.stat().st_size > 0, "OBJ file is empty"


def test_output_is_watertight():
    """Generated STL loads as a watertight mesh."""
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "piece.stl"
        generate(shape, out, format="stl")
        mesh = trimesh.load(str(out))
        assert mesh.is_watertight, "Loaded STL mesh is not watertight"


def test_irregular_shape_generates_valid_stl():
    """An irregular polygon outline produces a non-empty watertight STL."""
    shape = make_irregular_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "irregular.stl"
        generate(shape, out, format="stl")
        assert out.exists() and out.stat().st_size > 0
        mesh = trimesh.load(str(out))
        assert mesh.is_watertight, "Irregular shape STL is not watertight"


def test_unsupported_format_raises_error():
    """Unknown format string raises ModelGenerationError."""
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "piece.xyz"
        with pytest.raises(ModelGenerationError, match="Unsupported format"):
            generate(shape, out, format="xyz")


def test_empty_outline_raises_error():
    """An outline with fewer than 3 points raises ModelGenerationError."""
    outline = np.array([[0.0, 0.0], [1.0, 0.0]])  # only 2 points
    shape = MissingPieceShape(
        outline=outline,
        width_mm=1.0,
        height_mm=1.0,
        pixel_to_mm_scale=1.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "bad.stl"
        with pytest.raises(ModelGenerationError):
            generate(shape, out, format="stl")


def test_bevel_param_accepted():
    """bevel_mm=0.5 runs without error and produces a valid file."""
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "beveled.stl"
        generate(shape, out, format="stl", bevel_mm=0.5)
        assert out.exists() and out.stat().st_size > 0


def test_thickness_applied():
    """The extruded mesh height matches the requested thickness_mm."""
    thickness = 6.0
    shape = make_square_shape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "thick.stl"
        generate(shape, out, format="stl", thickness_mm=thickness)
        mesh = trimesh.load(str(out))
        # bounds[1][2] is the maximum Z coordinate (top of the extrusion)
        assert pytest.approx(mesh.bounds[1][2], abs=0.01) == thickness

"""Shared dataclasses for the missing-piece-gen pipeline."""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class EdgeType(Enum):
    """Classification of a puzzle piece edge."""

    TAB = "tab"    # convex protrusion into the missing slot
    BLANK = "blank"  # concave indentation away from the missing slot
    FLAT = "flat"  # straight border edge (piece is on the puzzle perimeter)


@dataclass
class TabGeometry:
    """Approximate geometry of a tab or blank feature on an edge."""

    position: float  # 0.0-1.0, normalized position along edge baseline
    width: float     # mm
    depth: float     # mm (positive = protrusion, negative = indentation)


@dataclass
class PieceRegion:
    """A detected puzzle piece and its spatial context."""

    piece_id: int
    crop: np.ndarray                                       # BGR image crop
    bounding_polygon: np.ndarray                           # 4x2 array, corner points in image-space
    inward_edges: list[str]                                # sides facing the missing slot: "top"|"right"|"bottom"|"left"
    slot_bounding_box: tuple[int, int, int, int] | None = None  # (x, y, w, h) of the missing slot


@dataclass
class EdgeProfile:
    """Extracted contour and classification of one edge facing the missing slot."""

    direction: str                    # "top", "right", "bottom", "left"
    contour: np.ndarray               # Nx2 array, ordered 2D points in image-space
    edge_type: EdgeType
    tab_geometry: TabGeometry | None = None


@dataclass
class MissingPieceShape:
    """2D outline of the inferred missing piece, ready for 3D extrusion."""

    outline: np.ndarray        # Nx2 array, closed polygon in mm-space
    width_mm: float
    height_mm: float
    pixel_to_mm_scale: float   # conversion factor used

# Architecture: missing-piece-gen

> End-to-end pipeline for generating a 3D-printable replacement puzzle piece
> from a photo of the surrounding pieces.

## Pipeline Overview

```
                        missing-piece-gen <image_path>
                                  |
                                  v
  +---------------------------------------------------------------+
  |  cli.py  — validates input, wires stages, writes output file  |
  +---------------------------------------------------------------+
       |              |               |               |
       v              v               v               v
  +-----------+  +------------+  +-----------+  +------------+
  | segment-  |  | edge_      |  | inference |  | model_gen  |
  | ation.py  |->| analysis.py|->|    .py    |->|    .py     |
  +-----------+  +------------+  +-----------+  +------------+
  photo ->       PieceRegion ->  EdgeProfile -> MissingPiece ->
  List[Piece     List[Edge       MissingPiece   STL/OBJ file
   Region]        Profile]        Shape
```

## Module Structure

```
missing_piece_gen/
  __init__.py            # Package root, exposes __version__
  cli.py                 # Click entry point — validates args, wires pipeline
  models.py              # Shared dataclasses (PieceRegion, EdgeProfile, etc.)
  errors.py              # PipelineError and stage-specific exceptions
  segmentation.py        # Stage 1: photo -> List[PieceRegion]
  edge_analysis.py       # Stage 2: PieceRegion -> List[EdgeProfile]
  inference.py           # Stage 3: List[EdgeProfile] -> MissingPieceShape
  model_gen.py           # Stage 4: MissingPieceShape -> 3D file on disk
```

### Module responsibilities

| Module | Issue | Input | Output |
|--------|-------|-------|--------|
| `cli.py` | #3 | CLI args (image path, --output, --format, --thickness) | Orchestrates pipeline, writes file |
| `segmentation.py` | #4 | BGR image (`np.ndarray`) | `list[PieceRegion]` |
| `edge_analysis.py` | #5 | `PieceRegion` | `list[EdgeProfile]` |
| `inference.py` | #6 | `list[EdgeProfile]` | `MissingPieceShape` |
| `model_gen.py` | #7 | `MissingPieceShape`, output path, format, thickness | File written to disk |

### Import rule

Only `cli.py` is allowed to import from more than one stage module. Stage modules
import only from `models.py` and `errors.py`, never from each other. This
prevents circular dependencies and keeps stages independently testable.

## Data Classes (models.py)

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np


class EdgeType(Enum):
    """Classification of a puzzle piece edge."""
    TAB = "tab"        # convex protrusion into the missing slot
    BLANK = "blank"    # concave indentation away from the missing slot
    FLAT = "flat"      # straight border edge (piece is on the puzzle perimeter)


@dataclass
class TabGeometry:
    """Approximate geometry of a tab or blank feature on an edge."""
    position: float    # 0.0-1.0, normalized position along edge baseline
    width: float       # mm
    depth: float       # mm (positive = protrusion, negative = indentation)


@dataclass
class PieceRegion:
    """A detected puzzle piece and its spatial context."""
    piece_id: int
    crop: np.ndarray                                      # BGR image crop
    bounding_polygon: np.ndarray                          # 4x2 array, corner points in image-space
    inward_edges: list[str]                               # sides facing the missing slot: "top"|"right"|"bottom"|"left"
    slot_bounding_box: tuple[int, int, int, int] | None = None  # (x, y, w, h) of the missing slot


@dataclass
class EdgeProfile:
    """Extracted contour and classification of one edge facing the missing slot."""
    direction: str                        # "top", "right", "bottom", "left"
    contour: np.ndarray                   # Nx2 array, ordered 2D points in image-space
    edge_type: EdgeType
    tab_geometry: TabGeometry | None = None


@dataclass
class MissingPieceShape:
    """2D outline of the inferred missing piece, ready for 3D extrusion."""
    outline: np.ndarray                   # Nx2 array, closed polygon in mm-space
    width_mm: float
    height_mm: float
    pixel_to_mm_scale: float              # conversion factor used
```

## Coordinate System

| Space | Origin | Units | Convention |
|-------|--------|-------|------------|
| Image-space | Top-left corner of the photo | pixels | OpenCV (row, col); x = right, y = down |
| Model-space (2D) | Bottom-left of the missing piece bounding box | mm | CAD convention; x = right, y = up |
| Model-space (3D) | Same as 2D origin, z=0 at bottom face | mm | Z-axis is extrusion direction (upward) |

### Pixel-to-mm conversion

Standard puzzle piece sizes vary by puzzle count:

| Puzzle size | Typical piece width |
|-------------|-------------------|
| 500 pieces  | ~30 mm |
| 1000 pieces | ~20 mm |
| 1500 pieces | ~17 mm |

The `pixel_to_mm_scale` factor is estimated during segmentation based on a
user-provided or auto-detected reference dimension. For Sprint 1, the user
provides `--piece-width-mm` (default: 20.0) and the scale is derived from the
detected piece bounding box width in pixels.

### Default parameters

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| Output file | `missing_piece.stl` | `--output` |
| Output format | STL | `--format stl\|obj` |
| Piece thickness | 4.0 mm | `--thickness` |
| Edge bevel/chamfer | 0.5 mm | `--bevel` |
| Piece width hint | 20.0 mm | `--piece-width-mm` |

## Error Handling

All user-facing errors raise `PipelineError` (defined in `errors.py`).
Stage-specific subclasses provide context:

```python
class PipelineError(Exception):
    """Base error for all pipeline failures."""

class DetectionError(PipelineError):
    """Image segmentation could not detect pieces or the missing slot."""

class EdgeExtractionError(PipelineError):
    """Edge profile extraction failed for a piece."""

class InferenceError(PipelineError):
    """Could not compute a valid missing piece shape."""

class ModelGenerationError(PipelineError):
    """3D model generation or export failed."""
```

`cli.py` catches `PipelineError` and prints a human-readable message with
non-zero exit code. Unexpected exceptions propagate with a full traceback.

## Pipeline Execution Flow

```
cli.py main():
  1. Parse and validate CLI arguments
  2. Load image from disk (cv2.imread)
  3. segmentation.segment(image) -> List[PieceRegion]
  4. For each PieceRegion:
       edge_analysis.extract_edges(piece) -> List[EdgeProfile]
  5. inference.infer_shape(all_edges, pixel_to_mm_scale) -> MissingPieceShape
  6. model_gen.generate(shape, output_path, format, thickness, bevel) -> None
  7. Print success message with output file path
```

## Testing Strategy

Each stage module has an independent test file under `tests/`:

```
tests/
  conftest.py               # Shared fixtures (sample images, mock data)
  test_cli.py               # CLI argument parsing, exit codes
  test_segmentation.py      # Piece detection on sample images
  test_edge_analysis.py     # Edge extraction from fixture PieceRegions
  test_inference.py         # Shape assembly from fixture EdgeProfiles
  test_model_gen.py         # 3D output validity (watertight check)
```

Stage boundaries are mocked in integration tests so each stage can be tested
independently.

## Feature Issue Mapping

| Issue | Module | Depends on |
|-------|--------|-----------|
| #3 CLI scaffolding | `cli.py`, `__init__.py`, `models.py`, `errors.py` | -- |
| #4 Image processing | `segmentation.py` | #2 (this doc), #3 (project setup) |
| #5 Edge extraction | `edge_analysis.py` | #4 |
| #6 Missing piece inference | `inference.py` | #5 |
| #7 3D model generation | `model_gen.py` | #1, #6 |

## ADR Reference

The architectural decision for this pipeline design is recorded in
[ADR-20260528: End-to-end pipeline module structure and data contracts](https://github.com/wilburb/missing-piece-gen/issues/9).

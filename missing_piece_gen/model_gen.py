"""Stage 4: Extrude the 2D missing piece outline into a 3D-printable solid."""
from pathlib import Path
import trimesh
from shapely.geometry import Polygon
from .models import MissingPieceShape
from .errors import ModelGenerationError


def generate(
    shape: MissingPieceShape,
    output_path: str | Path,
    format: str = "stl",
    thickness_mm: float = 4.0,
    bevel_mm: float = 0.5,
) -> None:
    """
    Extrude the 2D outline into a 3D solid and export to file.

    Args:
        shape: MissingPieceShape with 2D outline in mm-space.
        output_path: Path to write the output file.
        format: "stl" or "obj".
        thickness_mm: Extrusion height in mm (default 4.0).
        bevel_mm: Chamfer/offset applied to top and bottom faces (default 0.5).
                  If 0, no bevel is applied.
                  NOTE: Full chamfer geometry (separate top/bottom inset faces)
                  is a future enhancement. For Sprint 1, bevel_mm is accepted
                  and a simple buffer is applied to validate the parameter, but
                  the extruded mesh uses the original polygon at full height.

    Raises:
        ModelGenerationError: If the mesh cannot be generated or is not watertight.
    """
    # 1. Validate input
    if shape.outline is None or len(shape.outline) < 3:
        raise ModelGenerationError(
            f"Outline must have at least 3 points; got "
            f"{len(shape.outline) if shape.outline is not None else 0}."
        )

    # 2. Build Shapely polygon from the outline
    polygon = Polygon(shape.outline)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # fix minor self-intersections
    if not polygon.is_valid or polygon.area <= 0:
        raise ModelGenerationError("Invalid 2D outline — cannot create polygon.")

    # 3. Apply bevel (optional)
    # Sprint 1: compute the inset polygon to validate bevel_mm is sane, but
    # extrusion uses the base polygon at full height. Full chamfer geometry
    # (multi-layer extrusion with inset top/bottom caps) is a future enhancement.
    if bevel_mm > 0:
        top_polygon = polygon.buffer(-bevel_mm)
        if top_polygon.is_empty:
            top_polygon = polygon  # bevel too large, skip it
    else:
        top_polygon = polygon  # noqa: F841 (reserved for future chamfer use)

    # 4. Extrude to 3D mesh using trimesh
    try:
        mesh = trimesh.creation.extrude_polygon(polygon, height=thickness_mm)
    except Exception as exc:
        raise ModelGenerationError(f"trimesh extrusion failed: {exc}") from exc

    # 5. Validate watertightness
    if not mesh.is_watertight:
        mesh.fill_holes()
        if not mesh.is_watertight:
            raise ModelGenerationError("Generated mesh is not watertight.")

    # 6. Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt == "stl":
        mesh.export(str(output_path), file_type="stl")
    elif fmt == "obj":
        mesh.export(str(output_path), file_type="obj")
    else:
        raise ModelGenerationError(
            f"Unsupported format: {format!r}. Use 'stl' or 'obj'."
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ModelGenerationError("Output file was not written or is empty.")

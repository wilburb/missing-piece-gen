"""Stage 4: Extrude the 2D missing piece outline into a 3D-printable solid."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from .models import MissingPieceShape
from .errors import ModelGenerationError


def generate(
    shape: MissingPieceShape,
    output_path: str | Path,
    format: str = "stl",
    thickness_mm: float = 4.0,
    bevel_mm: float = 0.5,
) -> None:
    """Extrude the 2D outline into a 3D solid and export to file.

    Args:
        shape:        MissingPieceShape with 2D outline in mm-space.
        output_path:  Path to write the output file.
        format:       "stl" or "obj".
        thickness_mm: Extrusion height in mm (default 4.0).
        bevel_mm:     Width of the chamfer on the top edge in mm (default 0.5).
                      The chamfer slopes from the full outline at
                      z=(thickness-bevel) down to an inset outline at z=thickness,
                      mimicking the tapered top edge of a real puzzle piece.
                      Set to 0 to disable.

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
        polygon = polygon.buffer(0)
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if not polygon.is_valid or polygon.area <= 0:
        raise ModelGenerationError("Invalid 2D outline — cannot create polygon.")

    # 3. Extrude with optional top chamfer
    try:
        if bevel_mm > 0:
            mesh = _extrude_with_top_chamfer(polygon, thickness_mm, bevel_mm)
        else:
            mesh = trimesh.creation.extrude_polygon(polygon, height=thickness_mm)
    except Exception as exc:
        raise ModelGenerationError(f"Mesh generation failed: {exc}") from exc

    # 4. Validate watertightness
    if not mesh.is_watertight:
        mesh.fill_holes()
        if not mesh.is_watertight:
            raise ModelGenerationError("Generated mesh is not watertight.")

    # 5. Export
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


# ---------------------------------------------------------------------------
# Chamfer implementation
# ---------------------------------------------------------------------------

def _sample_ring(ring, n: int) -> np.ndarray:
    """Return n evenly-spaced points along a Shapely LinearRing."""
    length = ring.length
    ts = np.linspace(0.0, length, n, endpoint=False)
    pts = np.empty((n, 2), dtype=np.float64)
    for k, t in enumerate(ts):
        p = ring.interpolate(t)
        pts[k, 0] = p.x
        pts[k, 1] = p.y
    return pts


def _extrude_with_top_chamfer(
    polygon: Polygon,
    thickness_mm: float,
    bevel_mm: float,
    n_ring: int = 400,
) -> trimesh.Trimesh:
    """Build an extruded mesh with a chamfered top edge.

    Geometry (cross-section):
        ___________    ← top cap (inset polygon, z = thickness)
       /           \\   ← chamfer slope  (z = thickness-bevel → thickness)
      |             |  ← vertical side walls (z = 0 → thickness-bevel)
      |_____________|  ← bottom cap (full polygon, z = 0)

    All faces are built from the same sampled rings so the mesh has no
    junction gaps and merge_vertices + fix_normals produces a watertight solid.
    """
    bevel = min(bevel_mm, thickness_mm * 0.45)

    # Inset polygon for the chamfered top face
    poly_inset = polygon.buffer(-bevel)
    if isinstance(poly_inset, MultiPolygon):
        poly_inset = max(poly_inset.geoms, key=lambda p: p.area)
    if poly_inset.is_empty or not poly_inset.is_valid or poly_inset.area <= 0:
        return trimesh.creation.extrude_polygon(polygon, height=thickness_mm)

    z0 = 0.0
    z1 = thickness_mm - bevel   # bottom of chamfer / top of straight walls
    z2 = thickness_mm           # top of piece (inset top face)

    N = n_ring
    outer = _sample_ring(polygon.exterior, N)   # full outline ring
    inner = _sample_ring(poly_inset.exterior, N)  # inset ring

    # Align inner ring's start to the point closest to outer[0] to minimise
    # quad twist across the chamfer.
    shift = int(np.argmin(np.linalg.norm(inner - outer[0], axis=1)))
    inner = np.roll(inner, -shift, axis=0)

    # ── Vertices ─────────────────────────────────────────────────────────────
    # Index layout (3·N vertices total):
    #   [0 .. N-1]   outer ring at z0   (bottom cap + base of side walls)
    #   [N .. 2N-1]  outer ring at z1   (top of side walls + base of chamfer)
    #   [2N .. 3N-1] inner ring at z2   (top of chamfer + top cap)
    outer_z0 = np.column_stack([outer, np.full(N, z0)])
    outer_z1 = np.column_stack([outer, np.full(N, z1)])
    inner_z2 = np.column_stack([inner, np.full(N, z2)])
    verts = np.vstack([outer_z0, outer_z1, inner_z2]).astype(np.float64)

    # ── Faces ────────────────────────────────────────────────────────────────
    i_arr = np.arange(N, dtype=np.int64)
    j_arr = (i_arr + 1) % N

    # Side walls: outer ring from z0 → z1 (quad strip)
    sw1 = np.column_stack([i_arr,       j_arr,       N + j_arr])
    sw2 = np.column_stack([i_arr,       N + j_arr,   N + i_arr])

    # Chamfer walls: outer at z1 → inner at z2 (quad strip)
    cw1 = np.column_stack([N + i_arr,   N + j_arr,   2*N + j_arr])
    cw2 = np.column_stack([N + i_arr,   2*N + j_arr, 2*N + i_arr])

    wall_faces = np.vstack([sw1, sw2, cw1, cw2]).astype(np.int64)

    # Bottom cap: triangulate outer ring at z0.
    # Earcut on a CCW ring (Shapely standard) produces normals pointing +z.
    # The bottom face needs normals pointing -z, so reverse the winding.
    from mapbox_earcut import triangulate_float64
    ring_ends = np.array([N], dtype=np.uint32)
    bot_raw = triangulate_float64(
        outer.astype(np.float64), ring_ends
    ).reshape(-1, 3).astype(np.int64)
    bot_tri = bot_raw[:, [0, 2, 1]]  # flip → normals point -z (downward)

    # Top cap: triangulate inner ring at z2.
    # CCW earcut → normals +z — correct for the top face, no flip needed.
    top_tri = (
        triangulate_float64(inner.astype(np.float64), ring_ends)
        .reshape(-1, 3)
        .astype(np.int64)
        + 2 * N
    )

    all_faces = np.vstack([bot_tri, wall_faces, top_tri])

    # ── Assemble ──────────────────────────────────────────────────────────────
    mesh = trimesh.Trimesh(vertices=verts, faces=all_faces, process=True)

    if not mesh.is_watertight:
        mesh.fill_holes()

    return mesh

"""Stage 3: Infer the 2D outline of the missing puzzle piece from surrounding edge profiles."""
import numpy as np
from .models import EdgeProfile, EdgeType, MissingPieceShape
from .errors import InferenceError

_DIRECTIONS = ("top", "right", "bottom", "left")

# The direction label on an EdgeProfile describes which edge of the *surrounding*
# piece faces the slot.  The missing piece's edge is the opposite side:
#   - piece above  → its "bottom" edge → constrains missing piece's "top"
#   - piece below  → its "top" edge    → constrains missing piece's "bottom"
#   - piece left   → its "right" edge  → constrains missing piece's "left"
#   - piece right  → its "left" edge   → constrains missing piece's "right"
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

# Number of sampled points per synthesised edge segment.
_N_EDGE_PTS = 64


def infer_shape(
    edge_profiles: list[EdgeProfile],
    pixel_to_mm_scale: float = 1.0,
    piece_width_hint_mm: float = 20.0,
    slot_width_px: float | None = None,
    slot_height_px: float | None = None,
) -> MissingPieceShape:
    """
    Compute the 2D outline of the missing piece from surrounding EdgeProfiles.

    Each surrounding piece contributes an edge profile that is the complement
    of the missing piece's corresponding edge. The complement of a TAB is a BLANK
    and vice versa. Flat edges are mirrored as-is.

    Args:
        edge_profiles: List of EdgeProfile objects from surrounding pieces.
                      Each profile's direction indicates which edge of the
                      *surrounding* piece faces the slot.
        pixel_to_mm_scale: Conversion factor (mm per pixel).
        piece_width_hint_mm: Fallback estimated piece width in mm (used when
                             slot dimensions are not provided).
        slot_width_px: Width of the missing slot in pixels (authoritative).
        slot_height_px: Height of the missing slot in pixels (authoritative).

    Returns:
        MissingPieceShape with a closed 2D polygon outline in mm-space.

    Raises:
        InferenceError: If a valid closed shape cannot be assembled.
    """
    # --- 1. Group profiles by the missing piece direction they constrain ---
    # ep.direction is the surrounding piece's inward edge; _OPPOSITE gives
    # which side of the missing piece that edge constrains.
    profiles_by_dir: dict[str, EdgeProfile] = {}
    for ep in edge_profiles:
        missing_dir = _OPPOSITE.get(ep.direction, ep.direction)
        if missing_dir in _DIRECTIONS and missing_dir not in profiles_by_dir:
            profiles_by_dir[missing_dir] = ep

    # --- 2. Determine bounding-box size in pixels ---
    piece_size_px = piece_width_hint_mm / pixel_to_mm_scale
    width_px, height_px = _estimate_piece_size(profiles_by_dir, piece_size_px)
    # Prefer the authoritative slot dimensions when available.
    if slot_width_px is not None and slot_width_px >= 1.0:
        width_px = float(slot_width_px)
    if slot_height_px is not None and slot_height_px >= 1.0:
        height_px = float(slot_height_px)

    # --- 3. Build each edge segment in the piece's local coordinate frame ---
    # Local frame: origin at top-left corner, x right, y down (image convention)
    # Piece bounding box: (0,0) to (width_px, height_px)
    top_seg = _build_edge_segment(profiles_by_dir.get("top"), "top", width_px, height_px)
    right_seg = _build_edge_segment(profiles_by_dir.get("right"), "right", width_px, height_px)
    bottom_seg = _build_edge_segment(profiles_by_dir.get("bottom"), "bottom", width_px, height_px)
    left_seg = _build_edge_segment(profiles_by_dir.get("left"), "left", width_px, height_px)

    # --- 4. Assemble closed polygon ---
    # Top: left→right (y ≈ 0)
    # Right: top→bottom (x ≈ width_px)
    # Bottom: right→left (y ≈ height_px)
    # Left: bottom→top (x ≈ 0)
    outline_px = np.concatenate([top_seg, right_seg, bottom_seg, left_seg], axis=0)

    if not np.allclose(outline_px[0], outline_px[-1]):
        outline_px = np.vstack([outline_px, outline_px[0]])

    # --- 5. Convert to mm-space ---
    outline_mm = outline_px * pixel_to_mm_scale
    width_mm = width_px * pixel_to_mm_scale
    height_mm = height_px * pixel_to_mm_scale

    # --- 6. Validate ---
    area = _shoelace_area(outline_mm)
    if area <= 0:
        raise InferenceError(
            f"Assembled outline has non-positive area ({area:.4f} mm²); "
            "cannot produce a valid missing piece shape."
        )

    return MissingPieceShape(
        outline=outline_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        pixel_to_mm_scale=pixel_to_mm_scale,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_piece_size(
    profiles_by_dir: dict[str, EdgeProfile],
    fallback_px: float,
) -> tuple[float, float]:
    """Return (width_px, height_px) estimated from contour extents or fallback."""
    width_px = fallback_px
    height_px = fallback_px

    top = profiles_by_dir.get("top")
    bottom = profiles_by_dir.get("bottom")
    if top is not None and len(top.contour) >= 2:
        width_px = float(np.ptp(top.contour[:, 0]))
    elif bottom is not None and len(bottom.contour) >= 2:
        width_px = float(np.ptp(bottom.contour[:, 0]))

    left = profiles_by_dir.get("left")
    right = profiles_by_dir.get("right")
    if left is not None and len(left.contour) >= 2:
        height_px = float(np.ptp(left.contour[:, 1]))
    elif right is not None and len(right.contour) >= 2:
        height_px = float(np.ptp(right.contour[:, 1]))

    if width_px < 1.0:
        width_px = fallback_px
    if height_px < 1.0:
        height_px = fallback_px

    return width_px, height_px


def _gaussian_bump(t: np.ndarray, center: float, width_px: float) -> np.ndarray:
    """Return a Gaussian curve over *t* with peak 1.0 at *center*.

    sigma is set so the bump is near-zero outside a ±2-sigma band whose
    full-width equals *width_px*.
    """
    sigma = max(width_px / 4.0, 1.0)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _build_edge_segment(
    profile: EdgeProfile | None,
    direction: str,
    width_px: float,
    height_px: float,
) -> np.ndarray:
    """
    Build the edge segment for one side of the missing piece in local pixel coords.

    Uses the surrounding piece's TabGeometry to synthesise a smooth Gaussian-
    shaped tab or blank on the missing piece:
      - Surrounding TAB  → missing piece has a BLANK (indents inward)
      - Surrounding BLANK → missing piece has a TAB  (protrudes outward)
      - Surrounding FLAT  → missing piece has a FLAT edge

    Returns an (N, 2) array of points ordered to travel along the edge:
        top: L→R   right: T→B   bottom: R→L   left: B→T
    """
    if profile is None:
        return _flat_edge(direction, width_px, height_px)

    edge_type = profile.edge_type
    tab_geom = profile.tab_geometry

    if edge_type == EdgeType.FLAT or tab_geom is None:
        return _flat_edge(direction, width_px, height_px)

    n = _N_EDGE_PTS
    pos = tab_geom.position        # 0–1 along the edge
    depth = tab_geom.depth         # pixels
    tab_width = tab_geom.width     # pixels

    # Surrounding TAB → missing BLANK (positive inward_depth = dips in)
    # Surrounding BLANK → missing TAB (negative inward_depth = protrudes out)
    inward_depth = depth if edge_type == EdgeType.TAB else -depth

    if direction == "top":
        # Baseline y=0, travel left→right.
        x_pts = np.linspace(0.0, width_px, n)
        bump = _gaussian_bump(x_pts, pos * width_px, tab_width) * inward_depth
        # positive → dips down from top (BLANK); negative → protrudes up (TAB)
        return np.column_stack([x_pts, bump])

    elif direction == "right":
        # Baseline x=width_px, travel top→bottom.
        y_pts = np.linspace(0.0, height_px, n)
        bump = _gaussian_bump(y_pts, pos * height_px, tab_width) * inward_depth
        # positive → dips left (BLANK); negative → protrudes right (TAB)
        x_pts = width_px - bump
        return np.column_stack([x_pts, y_pts])

    elif direction == "bottom":
        # Baseline y=height_px, travel right→left.
        x_pts = np.linspace(width_px, 0.0, n)
        bump = _gaussian_bump(x_pts, pos * width_px, tab_width) * inward_depth
        # positive → dips up (BLANK); negative → protrudes down (TAB)
        y_pts = height_px - bump
        return np.column_stack([x_pts, y_pts])

    else:  # left
        # Baseline x=0, travel bottom→top.
        y_pts = np.linspace(height_px, 0.0, n)
        bump = _gaussian_bump(y_pts, pos * height_px, tab_width) * inward_depth
        # positive → dips right (BLANK); negative → protrudes left (TAB)
        x_pts = bump
        return np.column_stack([x_pts, y_pts])


def _flat_edge(direction: str, width_px: float, height_px: float) -> np.ndarray:
    """Return a 2-point straight-line segment for the given side of the bounding box."""
    if direction == "top":
        return np.array([[0.0, 0.0], [width_px, 0.0]])
    elif direction == "right":
        return np.array([[width_px, 0.0], [width_px, height_px]])
    elif direction == "bottom":
        return np.array([[width_px, height_px], [0.0, height_px]])
    else:  # left
        return np.array([[0.0, height_px], [0.0, 0.0]])


def _shoelace_area(pts: np.ndarray) -> float:
    """Return the signed area of a polygon using the shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(x)
    area = 0.0
    for i in range(n - 1):
        area += x[i] * y[i + 1] - x[i + 1] * y[i]
    area += x[-1] * y[0] - x[0] * y[-1]
    return abs(area) / 2.0

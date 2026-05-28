"""Stage 3: Infer the 2D outline of the missing puzzle piece from surrounding edge profiles."""
import numpy as np
from .models import EdgeProfile, MissingPieceShape, EdgeType
from .errors import InferenceError

_DIRECTIONS = ("top", "right", "bottom", "left")


def infer_shape(
    edge_profiles: list[EdgeProfile],
    pixel_to_mm_scale: float = 1.0,
    piece_width_hint_mm: float = 20.0,
) -> MissingPieceShape:
    """
    Compute the 2D outline of the missing piece from surrounding EdgeProfiles.

    Each surrounding piece contributes an edge profile that is the complement
    of the missing piece's corresponding edge. The complement of a TAB is a BLANK
    and vice versa. Flat edges are mirrored as-is.

    Args:
        edge_profiles: List of EdgeProfile objects from surrounding pieces.
                      Each profile's direction indicates which side of the missing
                      piece it constrains ("top", "right", "bottom", "left").
        pixel_to_mm_scale: Conversion factor (mm per pixel).
        piece_width_hint_mm: Estimated piece width in mm (used if no profiles given).

    Returns:
        MissingPieceShape with a closed 2D polygon outline in mm-space.

    Raises:
        InferenceError: If a valid closed shape cannot be assembled.
    """
    # --- 1. Group profiles by direction (use first per direction) ---
    profiles_by_dir: dict[str, EdgeProfile] = {}
    for ep in edge_profiles:
        if ep.direction in _DIRECTIONS and ep.direction not in profiles_by_dir:
            profiles_by_dir[ep.direction] = ep

    # Estimate piece width/height in pixels from contour extents or hint
    piece_size_px = piece_width_hint_mm / pixel_to_mm_scale
    width_px, height_px = _estimate_piece_size(profiles_by_dir, piece_size_px)

    # --- 2. Build each edge segment in the piece's local coordinate frame ---
    # Local frame: origin at top-left corner, x right, y down (image convention)
    # Piece bounding box: (0,0) to (width_px, height_px)
    top_seg = _build_edge_segment(profiles_by_dir.get("top"), "top", width_px, height_px)
    right_seg = _build_edge_segment(profiles_by_dir.get("right"), "right", width_px, height_px)
    bottom_seg = _build_edge_segment(profiles_by_dir.get("bottom"), "bottom", width_px, height_px)
    left_seg = _build_edge_segment(profiles_by_dir.get("left"), "left", width_px, height_px)

    # --- 3. Assemble closed polygon ---
    # Top: left→right (y ≈ 0)
    # Right: top→bottom (x ≈ width_px)
    # Bottom: right→left (y ≈ height_px)
    # Left: bottom→top (x ≈ 0)
    outline_px = np.concatenate([top_seg, right_seg, bottom_seg, left_seg], axis=0)

    # Close the polygon by appending the first point if needed
    if not np.allclose(outline_px[0], outline_px[-1]):
        outline_px = np.vstack([outline_px, outline_px[0]])

    # --- 4. Convert to mm-space ---
    outline_mm = outline_px * pixel_to_mm_scale

    width_mm = width_px * pixel_to_mm_scale
    height_mm = height_px * pixel_to_mm_scale

    # --- 5. Validate ---
    area = _shoelace_area(outline_mm)
    if area <= 0:
        raise InferenceError(
            f"Assembled outline has non-positive area ({area:.4f} mm²); "
            "cannot produce a valid missing piece shape."
        )

    # --- 6. Return ---
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
        width_px = float(np.ptp(top.contour[:, 0]))  # x-span
    elif bottom is not None and len(bottom.contour) >= 2:
        width_px = float(np.ptp(bottom.contour[:, 0]))

    left = profiles_by_dir.get("left")
    right = profiles_by_dir.get("right")
    if left is not None and len(left.contour) >= 2:
        height_px = float(np.ptp(left.contour[:, 1]))  # y-span
    elif right is not None and len(right.contour) >= 2:
        height_px = float(np.ptp(right.contour[:, 1]))

    # Guard against degenerate sizes
    if width_px < 1.0:
        width_px = fallback_px
    if height_px < 1.0:
        height_px = fallback_px

    return width_px, height_px


def _build_edge_segment(
    profile: EdgeProfile | None,
    direction: str,
    width_px: float,
    height_px: float,
) -> np.ndarray:
    """
    Build the edge segment for one side of the missing piece in local pixel coords.

    The surrounding piece's edge is the *complement* of the missing piece's edge:
      - Surrounding TAB  → missing piece has a BLANK on that side (contour dips inward)
      - Surrounding BLANK → missing piece has a TAB on that side (contour protrudes)
      - Surrounding FLAT  → missing piece has a FLAT edge

    The contour from the surrounding piece is already in the surrounding piece's
    local image-space. We mirror it to fit the missing piece's bounding box.

    Returns an (M, 2) array of points ordered to travel along the edge in the
    direction described in the module docstring (top: L→R, right: T→B,
    bottom: R→L, left: B→T).
    """
    if profile is None or len(profile.contour) < 2:
        return _flat_edge(direction, width_px, height_px)

    contour = profile.contour.astype(float)

    # Normalise contour to the 0..1 range along its primary axis, then
    # remap to the piece's bounding box so it sits at the correct side.
    if direction in ("top", "bottom"):
        # Primary axis = x (horizontal)
        x_min, x_max = contour[:, 0].min(), contour[:, 0].max()
        span = x_max - x_min
        if span < 1.0:
            return _flat_edge(direction, width_px, height_px)

        # Normalised x across the piece width
        x_norm = (contour[:, 0] - x_min) / span  # 0→1

        # The perpendicular deviation from the surrounding piece's baseline:
        # for top/bottom edges the baseline is roughly horizontal and
        # deviation is in y.  We need to mirror (complement) so a TAB on the
        # neighbour becomes a BLANK dip in the missing piece.
        y_mid = contour[:, 1].mean()
        y_dev = contour[:, 1] - y_mid   # deviation from mean
        y_dev_complement = -y_dev       # flip to get complement

        # Sort by x
        order = np.argsort(x_norm)
        x_norm = x_norm[order]
        y_dev_complement = y_dev_complement[order]

        if direction == "top":
            # Edge runs along y=0 in the missing piece frame.
            # Positive y_dev_complement means protrusion upward → clamp to stay
            # within a reasonable band so the outline doesn't leave the piece.
            x_pts = x_norm * width_px
            # Mirror: surrounding TAB protrudes toward the slot (y decreases);
            # missing piece edge curves inward (y increases from 0).
            y_pts = -y_dev_complement  # positive = inward (down from top edge)
            pts = np.column_stack([x_pts, y_pts])
            # Travel left → right
            pts = pts[np.argsort(pts[:, 0])]

        else:  # bottom
            # Edge runs along y=height_px.
            x_pts = x_norm * width_px
            # Travel right → left so we sort descending x
            y_pts = height_px + y_dev_complement
            pts = np.column_stack([x_pts, y_pts])
            pts = pts[np.argsort(-pts[:, 0])]

    else:
        # direction in ("left", "right")
        # Primary axis = y (vertical)
        y_min, y_max = contour[:, 1].min(), contour[:, 1].max()
        span = y_max - y_min
        if span < 1.0:
            return _flat_edge(direction, width_px, height_px)

        y_norm = (contour[:, 1] - y_min) / span  # 0→1
        x_mid = contour[:, 0].mean()
        x_dev = contour[:, 0] - x_mid
        x_dev_complement = -x_dev

        order = np.argsort(y_norm)
        y_norm = y_norm[order]
        x_dev_complement = x_dev_complement[order]

        if direction == "right":
            # Edge runs along x=width_px; travel top → bottom
            y_pts = y_norm * height_px
            x_pts = width_px + x_dev_complement
            pts = np.column_stack([x_pts, y_pts])
            pts = pts[np.argsort(pts[:, 1])]

        else:  # left
            # Edge runs along x=0; travel bottom → top (descending y)
            y_pts = y_norm * height_px
            x_pts = -x_dev_complement
            pts = np.column_stack([x_pts, y_pts])
            pts = pts[np.argsort(-pts[:, 1])]

    return pts


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
    # Close the polygon
    area += x[-1] * y[0] - x[0] * y[-1]
    return abs(area) / 2.0

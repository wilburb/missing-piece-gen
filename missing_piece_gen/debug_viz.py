"""Debug visualization helpers — annotated images for troubleshooting the pipeline."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .models import EdgeProfile, EdgeType, MissingPieceShape, PieceRegion

# Colours (BGR)
_RED = (0, 0, 220)
_GREEN = (0, 200, 0)
_BLUE = (200, 100, 0)
_ORANGE = (0, 140, 255)
_MAGENTA = (200, 0, 200)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_LIGHT_BLUE = (230, 210, 180)

_EDGE_TYPE_COLOUR = {
    EdgeType.TAB: (0, 180, 0),      # green
    EdgeType.BLANK: (0, 0, 200),    # red
    EdgeType.FLAT: (150, 150, 150), # grey
}


def save_detection_image(
    image: np.ndarray,
    pieces: list[PieceRegion],
    all_edges: list[EdgeProfile],
    output_path: str | Path,
) -> Path:
    """Save an annotated copy of the input image showing what was detected.

    Annotations:
    - Red rectangle: detected missing-slot bounding box
    - Green polygon: each detected surrounding piece
    - Colour-coded edge-type label on each inward-facing edge:
        green = TAB, red = BLANK, grey = FLAT
    """
    output_path = Path(output_path)
    img = image.copy()

    # --- Slot bounding box ---
    slot_box = next(
        (p.slot_bounding_box for p in pieces if p.slot_bounding_box is not None),
        None,
    )
    if slot_box is not None:
        sx, sy, sw, sh = slot_box
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), _RED, 3)
        cv2.putText(img, "SLOT", (sx + 4, sy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2, cv2.LINE_AA)

    # Re-build per-piece edge list in order (all_edges is produced by iterating pieces).
    edges_per_piece: dict[int, list[EdgeProfile]] = {}
    ep_iter = iter(all_edges)
    for piece in pieces:
        edges_per_piece[piece.piece_id] = []
        for _ in piece.inward_edges:
            try:
                edges_per_piece[piece.piece_id].append(next(ep_iter))
            except StopIteration:
                break

    # --- Each piece ---
    piece_colours = [_GREEN, _ORANGE, _MAGENTA, _BLUE]
    for piece in pieces:
        colour = piece_colours[piece.piece_id % len(piece_colours)]

        if piece.bounding_polygon is not None and len(piece.bounding_polygon) >= 2:
            poly = piece.bounding_polygon.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [poly], isClosed=True, color=colour, thickness=2)

            cx = int(np.mean(piece.bounding_polygon[:, 0]))
            cy = int(np.mean(piece.bounding_polygon[:, 1]))
            bx = int(piece.bounding_polygon[:, 0].min())
            by = int(piece.bounding_polygon[:, 1].min())
            bw = int(piece.bounding_polygon[:, 0].max()) - bx
            bh = int(piece.bounding_polygon[:, 1].max()) - by

            # Piece ID
            cv2.putText(img, f"#{piece.piece_id}", (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

            # Edge-type labels placed near the relevant side of the bounding box
            side_profiles = edges_per_piece.get(piece.piece_id, [])
            for ep in side_profiles:
                et_colour = _EDGE_TYPE_COLOUR.get(ep.edge_type, _BLACK)
                label = f"{ep.direction}:{ep.edge_type.value}"
                depth_str = (
                    f"(d={ep.tab_geometry.depth:.0f}px)" if ep.tab_geometry else ""
                )
                full_label = f"{label} {depth_str}"

                # Anchor the label near the relevant edge of the piece bbox
                if ep.direction == "top":
                    lx, ly = bx, by - 6
                elif ep.direction == "bottom":
                    lx, ly = bx, by + bh + 16
                elif ep.direction == "left":
                    lx, ly = max(0, bx - 10), cy - 20
                else:  # right
                    lx, ly = bx + bw + 4, cy - 20

                # Background rectangle for readability
                (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                              _WHITE, cv2.FILLED)
                cv2.putText(img, full_label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, et_colour, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


def save_shape_image(
    shape: MissingPieceShape,
    profiles_by_dir: dict[str, EdgeProfile],
    output_path: str | Path,
) -> Path:
    """Save a 2D rendering of the inferred missing piece outline.

    The outline is drawn filled (light blue) with a black border.
    Each side is labelled with the edge type (TAB / BLANK / FLAT) and, where
    available, the tab depth in mm.
    """
    output_path = Path(output_path)
    canvas = 900
    margin = 80

    outline = shape.outline.copy()
    x_vals, y_vals = outline[:, 0], outline[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    w_mm = x_max - x_min
    h_mm = y_max - y_min

    if w_mm < 0.001 or h_mm < 0.001:
        scale = 1.0
    else:
        scale = (canvas - 2 * margin) / max(w_mm, h_mm)

    img = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255

    # Scale outline to canvas coordinates
    pts = np.column_stack([
        (x_vals - x_min) * scale + margin,
        (y_vals - y_min) * scale + margin,
    ]).astype(np.int32)

    cv2.fillPoly(img, [pts], _LIGHT_BLUE)
    cv2.polylines(img, [pts], isClosed=True, color=_BLACK, thickness=2)

    # Edge-type labels
    _label_edge(img, profiles_by_dir.get("top"), "top",
                margin, margin,
                int(margin + w_mm * scale), margin,
                shape.pixel_to_mm_scale)
    _label_edge(img, profiles_by_dir.get("right"), "right",
                int(margin + w_mm * scale), margin,
                int(margin + w_mm * scale), int(margin + h_mm * scale),
                shape.pixel_to_mm_scale)
    _label_edge(img, profiles_by_dir.get("bottom"), "bottom",
                int(margin + w_mm * scale), int(margin + h_mm * scale),
                margin, int(margin + h_mm * scale),
                shape.pixel_to_mm_scale)
    _label_edge(img, profiles_by_dir.get("left"), "left",
                margin, int(margin + h_mm * scale),
                margin, margin,
                shape.pixel_to_mm_scale)

    # Dimensions
    dim_label = f"{shape.width_mm:.1f} mm  x  {shape.height_mm:.1f} mm"
    cv2.putText(img, dim_label, (margin, canvas - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BLACK, 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


def _label_edge(
    img: np.ndarray,
    profile: EdgeProfile | None,
    direction: str,
    x1: int, y1: int,
    x2: int, y2: int,
    pixel_to_mm_scale: float,
) -> None:
    """Draw an edge-type annotation near the midpoint of a side."""
    if profile is None:
        label = f"{direction}: (no profile)"
        colour = (180, 180, 180)
    else:
        et = profile.edge_type
        colour = _EDGE_TYPE_COLOUR.get(et, _BLACK)
        depth_str = ""
        if profile.tab_geometry is not None:
            depth_mm = profile.tab_geometry.depth * pixel_to_mm_scale
            depth_str = f"  depth={depth_mm:.1f}mm"
        label = f"{direction}: {et.value}{depth_str}"

    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2

    # Offset label away from the shape edge
    offset = 24
    if direction == "top":
        lx, ly = mx - 60, my - offset
    elif direction == "bottom":
        lx, ly = mx - 60, my + offset + 12
    elif direction == "left":
        lx, ly = max(0, mx - 140), my
    else:  # right
        lx, ly = mx + 6, my

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 4),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def save_edge_crops(
    pieces: list[PieceRegion],
    all_edges: list[EdgeProfile],
    output_dir: str | Path,
) -> list[Path]:
    """Save a cropped image for every inward-facing edge showing the ROI that
    edge_analysis processed, overlaid with the extracted contour points and
    the detected edge type.

    Files are named: ``piece<id>_<direction>_<edge_type>.jpg``

    Returns a list of paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ROI_FRAC = 0.30  # must match edge_analysis._EDGE_ROI_FRACTION

    paths: list[Path] = []
    ep_iter = iter(all_edges)
    for piece in pieces:
        for direction in piece.inward_edges:
            try:
                ep = next(ep_iter)
            except StopIteration:
                break

            if piece.crop is None or piece.crop.size == 0:
                continue

            crop = piece.crop.copy()
            h, w = crop.shape[:2]

            # Compute the same ROI slice as edge_analysis
            frac = _ROI_FRAC
            if direction == "top":
                roi_h = max(1, int(h * frac))
                roi = crop[:roi_h, :]
            elif direction == "bottom":
                roi_h = max(1, int(h * frac))
                roi = crop[h - roi_h:, :]
            elif direction == "left":
                roi_w = max(1, int(w * frac))
                roi = crop[:, :roi_w]
            elif direction == "right":
                roi_w = max(1, int(w * frac))
                roi = crop[:, w - roi_w:]
            else:
                roi = crop

            # Scale up small ROIs for visibility
            min_side = 200
            rh, rw = roi.shape[:2]
            scale = max(1, min_side // max(rh, rw, 1))
            if scale > 1:
                roi = cv2.resize(roi, (rw * scale, rh * scale),
                                 interpolation=cv2.INTER_NEAREST)

            # Overlay contour points (translated back to ROI-local coords)
            if ep.contour is not None and len(ep.contour) >= 2:
                # Determine the ROI origin in crop coords
                if direction == "bottom":
                    roi_y0 = h - max(1, int(h * frac))
                    roi_x0 = 0
                elif direction == "right":
                    roi_x0 = w - max(1, int(w * frac))
                    roi_y0 = 0
                else:
                    roi_x0, roi_y0 = 0, 0

                ct_colour = _EDGE_TYPE_COLOUR.get(ep.edge_type, _BLACK)
                for pt in ep.contour:
                    px = int((pt[0] - roi_x0) * scale)
                    py = int((pt[1] - roi_y0) * scale)
                    rh_s, rw_s = roi.shape[:2]
                    if 0 <= px < rw_s and 0 <= py < rh_s:
                        cv2.circle(roi, (px, py), max(2, scale), ct_colour, -1)

            # Label
            et_colour = _EDGE_TYPE_COLOUR.get(ep.edge_type, _BLACK)
            depth_str = (
                f" d={ep.tab_geometry.depth:.0f}px" if ep.tab_geometry else ""
            )
            label = f"#{piece.piece_id} {direction}: {ep.edge_type.value}{depth_str}"
            cv2.putText(roi, label, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 2, cv2.LINE_AA)
            cv2.putText(roi, label, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, et_colour, 1, cv2.LINE_AA)

            fname = f"piece{piece.piece_id}_{direction}_{ep.edge_type.value}.jpg"
            out_path = output_dir / fname
            cv2.imwrite(str(out_path), roi)
            paths.append(out_path)

    return paths

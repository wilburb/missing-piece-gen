"""CLI entry point for missing-piece-gen."""

import sys
import os
from pathlib import Path

import click
import cv2
import numpy as np

from missing_piece_gen import __version__
from missing_piece_gen.errors import PipelineError
from missing_piece_gen import segmentation, edge_analysis, inference, model_gen
from missing_piece_gen import debug_viz
from missing_piece_gen.inference import _OPPOSITE

OUTPUT_FORMATS = ("stl", "obj")

# Default value used when a PieceRegion has no bounding polygon.
_FALLBACK_PIECE_WIDTH_PX = 100.0


@click.command()
@click.version_option(version=__version__, prog_name="missing-piece-gen")
@click.argument("image_path", metavar="<image_path>")
@click.option(
    "--output",
    "-o",
    default="missing_piece.stl",
    show_default=True,
    metavar="<path>",
    help="Path for the output 3D model file.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="stl",
    show_default=True,
    type=click.Choice(OUTPUT_FORMATS, case_sensitive=False),
    help="Output file format.",
)
@click.option(
    "--thickness",
    default=4.0,
    show_default=True,
    type=float,
    help="Piece thickness in mm.",
)
@click.option(
    "--bevel",
    default=0.5,
    show_default=True,
    type=float,
    help="Edge bevel/chamfer in mm.",
)
@click.option(
    "--piece-width-mm",
    default=20.0,
    show_default=True,
    type=float,
    help="Estimated real-world width of a puzzle piece in mm (used for pixel-to-mm scale).",
)
@click.option(
    "--debug-dir",
    default=None,
    metavar="<dir>",
    help=(
        "Directory to write debug images (default: same directory as --output). "
        "Saves debug_detection.jpg (annotated input showing detected pieces, slot, and "
        "per-edge TAB/BLANK/FLAT classifications) and debug_shape.jpg (inferred 2D "
        "outline with edge-type labels and dimensions)."
    ),
)
@click.option(
    "--orange-ref",
    default=None,
    metavar="<path>",
    help=(
        "Path to a reference photo of just the orange backdrop. "
        "When supplied, the tool auto-calibrates the HSV detection range from that "
        "image instead of using the built-in defaults."
    ),
)
def main(
    image_path: str,
    output: str,
    output_format: str,
    thickness: float,
    bevel: float,
    piece_width_mm: float,
    debug_dir: str | None,
    orange_ref: str | None,
) -> None:
    """Generate a 3D-printable missing puzzle piece from IMAGE_PATH.

    The pipeline detects the missing region in IMAGE_PATH, builds a 3D
    model of the piece, and writes the result to the output file.
    """
    try:
        # 1. Validate input path
        if not os.path.exists(image_path):
            click.echo(
                f"Error: image path does not exist: {image_path}", err=True
            )
            sys.exit(1)

        # 2. Load image from disk
        click.echo(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise PipelineError(
                f"Could not read image file: {image_path}. "
                "Ensure the file is a supported image format."
            )

        # 3. Optional: calibrate orange HSV range from a reference photo
        orange_hsv_low = None
        orange_hsv_high = None
        if orange_ref is not None:
            ref_image = cv2.imread(orange_ref)
            if ref_image is None:
                click.echo(
                    f"Warning: could not read orange-ref image: {orange_ref}. "
                    "Falling back to built-in HSV defaults.",
                    err=True,
                )
            else:
                low, high = segmentation.calibrate_orange_hsv(ref_image)
                orange_hsv_low = low
                orange_hsv_high = high
                click.echo(
                    f"  [orange-ref] Calibrated HSV range: "
                    f"{low.tolist()} – {high.tolist()}"
                )

        # 4. Segment surrounding pieces
        click.echo("Detecting missing region...")
        pieces = segmentation.segment(image, orange_hsv_low=orange_hsv_low, orange_hsv_high=orange_hsv_high)
        click.echo(f"  Found {len(pieces)} surrounding piece(s).")

        # 5. Extract edge profiles for each piece
        click.echo("Extracting edge profiles...")
        all_edges = []
        for piece in pieces:
            profiles = edge_analysis.extract_edges(piece)
            all_edges.extend(profiles)
            for ep in profiles:
                depth_str = (
                    f", depth={ep.tab_geometry.depth:.1f}px"
                    if ep.tab_geometry else ""
                )
                click.echo(
                    f"  Piece #{piece.piece_id} {ep.direction!r:8s} "
                    f"→ {ep.edge_type.value}{depth_str}"
                )

        # 6. Compute pixel-to-mm scale from the average bounding-box width of pieces.
        widths_px = []
        for piece in pieces:
            if piece.bounding_polygon is not None and len(piece.bounding_polygon) >= 2:
                w = float(np.ptp(piece.bounding_polygon[:, 0]))
                if w > 0:
                    widths_px.append(w)
        avg_piece_width_px = float(np.mean(widths_px)) if widths_px else _FALLBACK_PIECE_WIDTH_PX
        pixel_to_mm_scale = piece_width_mm / avg_piece_width_px
        click.echo(
            f"  Pixel-to-mm scale: {pixel_to_mm_scale:.4f} mm/px "
            f"(avg piece width {avg_piece_width_px:.0f} px = {piece_width_mm:.1f} mm)"
        )

        # 7. Estimate slot dimensions from the surrounding pieces themselves.
        #
        # The slot bounding box from segmentation is unreliable in real photos
        # (shadows, dark artwork, or background can be detected instead of the gap).
        # Derive dimensions from neighbours instead:
        #   - pieces above/below the slot → their bounding width ≈ slot width
        #   - pieces left/right of the slot → their bounding height ≈ slot height
        slot_width_candidates: list[float] = []
        slot_height_candidates: list[float] = []
        for piece in pieces:
            if piece.bounding_polygon is None or len(piece.bounding_polygon) < 2:
                continue
            pw = float(np.ptp(piece.bounding_polygon[:, 0]))
            ph = float(np.ptp(piece.bounding_polygon[:, 1]))
            for edge_dir in piece.inward_edges:
                if edge_dir in ("top", "bottom") and pw > 0:
                    slot_width_candidates.append(pw)
                elif edge_dir in ("left", "right") and ph > 0:
                    slot_height_candidates.append(ph)

        slot_width_px: float | None = (
            float(np.mean(slot_width_candidates)) if slot_width_candidates else None
        )
        slot_height_px: float | None = (
            float(np.mean(slot_height_candidates)) if slot_height_candidates else None
        )
        if slot_width_px is None:
            slot_width_px = avg_piece_width_px if avg_piece_width_px > 0 else None
        if slot_height_px is None:
            slot_height_px = avg_piece_width_px if avg_piece_width_px > 0 else None

        if slot_width_px and slot_height_px:
            click.echo(
                f"  Piece size estimate: "
                f"{slot_width_px * pixel_to_mm_scale:.1f} × "
                f"{slot_height_px * pixel_to_mm_scale:.1f} mm"
            )

        # 8. Infer the missing piece shape
        click.echo("Inferring missing piece shape...")
        shape = inference.infer_shape(
            all_edges,
            pixel_to_mm_scale,
            slot_width_px=slot_width_px,
            slot_height_px=slot_height_px,
        )
        click.echo(
            f"  Shape: {shape.width_mm:.1f} × {shape.height_mm:.1f} mm"
        )

        # 9. Debug images — always produced; --debug-dir overrides the directory.
        # Default: same directory as the output file.
        ddir = Path(debug_dir) if debug_dir else Path(output).parent
        det_path = debug_viz.save_detection_image(
            image, pieces, all_edges, ddir / "debug_detection.jpg"
        )
        click.echo(f"  Debug detection image: {det_path}")

        # Build profiles_by_dir for the shape image
        profiles_by_dir: dict = {}
        for ep in all_edges:
            missing_dir = _OPPOSITE.get(ep.direction, ep.direction)
            if missing_dir not in profiles_by_dir:
                profiles_by_dir[missing_dir] = ep
        shape_path = debug_viz.save_shape_image(
            shape, profiles_by_dir, ddir / "debug_shape.jpg"
        )
        click.echo(f"  Debug shape image:     {shape_path}")

        edge_paths = debug_viz.save_edge_crops(pieces, all_edges, ddir / "debug_edges")
        if edge_paths:
            click.echo(f"  Debug edge crops:      {ddir / 'debug_edges'}/ ({len(edge_paths)} images)")

        # 10. Generate and write the 3D model
        click.echo(f"Generating 3D model (format={output_format})...")
        model_gen.generate(
            shape,
            output_path=output,
            format=output_format,
            thickness_mm=thickness,
            bevel_mm=bevel,
        )

        # 11. Success
        click.echo(f"Done. Output written to: {output}")

    except PipelineError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

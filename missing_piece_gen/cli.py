"""CLI entry point for missing-piece-gen."""

import sys
import os

import click
import cv2
import numpy as np

from missing_piece_gen import __version__
from missing_piece_gen.errors import PipelineError
from missing_piece_gen import segmentation, edge_analysis, inference, model_gen

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
def main(
    image_path: str,
    output: str,
    output_format: str,
    thickness: float,
    bevel: float,
    piece_width_mm: float,
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

        # 3. Segment surrounding pieces
        click.echo("Detecting missing region...")
        pieces = segmentation.segment(image)

        # 4. Extract edge profiles for each piece
        click.echo("Extracting edge profiles...")
        all_edges = []
        for piece in pieces:
            profiles = edge_analysis.extract_edges(piece)
            all_edges.extend(profiles)

        # Compute pixel-to-mm scale from the average bounding-box width of pieces.
        widths_px = []
        for piece in pieces:
            if piece.bounding_polygon is not None and len(piece.bounding_polygon) >= 2:
                w = float(np.ptp(piece.bounding_polygon[:, 0]))
                if w > 0:
                    widths_px.append(w)
        avg_piece_width_px = float(np.mean(widths_px)) if widths_px else _FALLBACK_PIECE_WIDTH_PX
        pixel_to_mm_scale = piece_width_mm / avg_piece_width_px

        # 5. Estimate slot dimensions from the surrounding pieces themselves.
        #
        # The slot bounding box from segmentation can be unreliable in real photos
        # (shadows, dark puzzle artwork, or table background may be detected
        # instead of the actual gap).  Instead, derive dimensions from neighbours:
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

        click.echo("Inferring missing piece shape...")
        shape = inference.infer_shape(
            all_edges,
            pixel_to_mm_scale,
            slot_width_px=slot_width_px,
            slot_height_px=slot_height_px,
        )

        # 6. Generate and write the 3D model
        click.echo(f"Generating 3D model (format={output_format})...")
        model_gen.generate(
            shape,
            output_path=output,
            format=output_format,
            thickness_mm=thickness,
            bevel_mm=bevel,
        )

        # 7. Success
        click.echo(f"Done. Output written to: {output}")

    except PipelineError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

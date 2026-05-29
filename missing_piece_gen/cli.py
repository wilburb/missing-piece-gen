"""CLI entry point for missing-piece-gen."""

import sys
import os
from pathlib import Path

import click
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.interpolate import splprep, splev

from missing_piece_gen import __version__
from missing_piece_gen.errors import PipelineError
from missing_piece_gen import segmentation, model_gen
from missing_piece_gen.models import MissingPieceShape

OUTPUT_FORMATS = ("stl", "obj")

# Tight HSV range used to locate the orange seed point (Archive analyze.py values)
_SEED_HSV_LOW  = np.array([0,  150, 150], dtype=np.uint8)
_SEED_HSV_HIGH = np.array([20, 255, 255], dtype=np.uint8)

# Default scale when no ruler is present and piece-width-mm cannot be applied
_DEFAULT_PX_PER_MM = 16.0


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
    default=None,
    show_default=True,
    type=float,
    help=(
        "Real-world width of the missing slot in mm. "
        "Used to calibrate pixel scale when no ruler is present. "
        "If omitted, the default of 16 px/mm is used."
    ),
)
@click.option(
    "--debug-dir",
    default=None,
    metavar="<dir>",
    help=(
        "Directory to write debug images. "
        "Saves debug_detection.jpg (hole contour overlaid on input) and "
        "debug_shape.jpg (2D outline rendering)."
    ),
)
@click.option(
    "--orange-ref",
    default=None,
    metavar="<path>",
    help=(
        "Path to a reference photo of just the orange marker. "
        "Auto-calibrates the HSV detection range from that image."
    ),
)
@click.option(
    "--ruler",
    is_flag=True,
    default=False,
    help="Detect a mm ruler at the left edge of the image for accurate px/mm calibration.",
)
def main(
    image_path: str,
    output: str,
    output_format: str,
    thickness: float,
    bevel: float,
    piece_width_mm: float | None,
    debug_dir: str | None,
    orange_ref: str | None,
    ruler: bool,
) -> None:
    """Generate a 3D-printable missing puzzle piece from IMAGE_PATH.

    Place a piece of orange paper inside the missing slot before photographing.
    The tool locates the orange, traces the hole boundary using the Archive
    not-white-complement strategy, fits a smooth B-spline, and extrudes the
    result into a watertight 3D mesh.
    """
    try:
        # ── 1. Validate input ─────────────────────────────────────────────────
        if not os.path.exists(image_path):
            click.echo(f"Error: image path does not exist: {image_path}", err=True)
            sys.exit(1)

        # ── 2. Load image ─────────────────────────────────────────────────────
        click.echo(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise PipelineError(
                f"Could not read image file: {image_path}. "
                "Ensure the file is a supported image format."
            )
        img_h, img_w = image.shape[:2]
        click.echo(f"  Image size: {img_w}×{img_h} px")

        # ── 3. Optional: calibrate orange HSV from reference photo ────────────
        orange_hsv_low  = _SEED_HSV_LOW.copy()
        orange_hsv_high = _SEED_HSV_HIGH.copy()
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
                orange_hsv_low  = low
                orange_hsv_high = high
                click.echo(
                    f"  [orange-ref] Calibrated HSV range: "
                    f"{low.tolist()} – {high.tolist()}"
                )

        # ── 4. Optional: ruler-based px/mm calibration ────────────────────────
        ruler_px_per_mm: float | None = None
        if ruler:
            gray_for_ruler = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ruler_px_per_mm = segmentation.calibrate_from_ruler(gray_for_ruler)
            if ruler_px_per_mm is not None:
                click.echo(f"  [ruler] Calibrated: {ruler_px_per_mm:.2f} px/mm")
            else:
                click.echo(
                    "  [ruler] Warning: could not detect ruler ticks.",
                    err=True,
                )

        # ── 5. Detect hole — Archive approach ─────────────────────────────────
        click.echo("Detecting missing region...")

        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step A: find orange seed centroid (tightest HSV range)
        seed_raw = cv2.inRange(img_hsv, orange_hsv_low, orange_hsv_high)
        n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(seed_raw, 8)
        if n_s <= 1:
            raise PipelineError(
                "No orange region found in the image. "
                "Place orange paper in the missing slot before photographing."
            )
        big_s = 1 + int(np.argmax(st_s[1:, cv2.CC_STAT_AREA]))
        m0 = cv2.moments((lbl_s == big_s).astype(np.uint8))
        if m0["m00"] == 0:
            raise PipelineError("Orange region has zero area.")
        seed_x = int(m0["m10"] / m0["m00"])
        seed_y = int(m0["m01"] / m0["m00"])
        click.echo(f"  Orange seed: ({seed_x}, {seed_y})")

        # Step B: medianBlur seals text on white puzzle backs (ksize=41 ≈ 20px strokes)
        gray_med     = cv2.medianBlur(gray, 41)
        white_sealed = (gray_med > 160).astype(np.uint8) * 255
        not_white    = cv2.bitwise_not(white_sealed)

        # Step C: connected component from seed = exactly the hole region
        _, labels_nw = cv2.connectedComponents(not_white, connectivity=8)
        hole_label   = labels_nw[seed_y, seed_x]
        if hole_label == 0:
            raise PipelineError(
                "The orange seed falls inside a white region after median blur. "
                "The orange marker may be too small, too bright, or occluded."
            )
        blob = (labels_nw == hole_label).astype(np.uint8) * 255

        # Step D: fill_holes → morph_close(9×9) → fill_holes → gaussian(σ=12)
        blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, close_k)
        blob = binary_fill_holes(blob.astype(bool)).astype(np.uint8) * 255
        blob_f    = gaussian_filter(blob.astype(np.float32) / 255.0, sigma=12)
        hole_mask = (blob_f > 0.5).astype(np.uint8) * 255

        # Step E: extract contour from cleaned mask
        contours, _ = cv2.findContours(
            hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise PipelineError("Could not extract hole contour after mask cleanup.")
        pts_raw = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        n_pts = len(pts_raw)
        click.echo(f"  Contour points (raw): {n_pts}")

        # Step F: fit periodic B-spline (Archive: s=N*20, per=True, 600 samples)
        if n_pts < 10:
            raise PipelineError(
                f"Hole contour has only {n_pts} points; image may be too small."
            )
        tck, _ = splprep([pts_raw[:, 0], pts_raw[:, 1]], s=n_pts * 20, per=True, k=3)
        u_fine = np.linspace(0, 1, 600, endpoint=False)
        sx, sy  = splev(u_fine, tck)
        pts_smooth = np.column_stack([sx, sy])
        click.echo(f"  Spline output points: {len(pts_smooth)}")

        # ── 6. Pixel-to-mm scale ──────────────────────────────────────────────
        x_min, y_min = pts_smooth.min(axis=0)
        x_max, y_max = pts_smooth.max(axis=0)
        hole_width_px  = x_max - x_min
        hole_height_px = y_max - y_min

        if ruler_px_per_mm is not None:
            px_per_mm = ruler_px_per_mm
            click.echo(f"  Scale: {px_per_mm:.2f} px/mm (ruler)")
        elif piece_width_mm is not None and hole_width_px > 0:
            px_per_mm = hole_width_px / piece_width_mm
            click.echo(
                f"  Scale: {px_per_mm:.2f} px/mm "
                f"(hole {hole_width_px:.0f} px = {piece_width_mm:.1f} mm)"
            )
        else:
            px_per_mm = _DEFAULT_PX_PER_MM
            click.echo(
                f"  Scale: {px_per_mm:.2f} px/mm (default — use --piece-width-mm for accuracy)"
            )

        mm_per_px  = 1.0 / px_per_mm
        width_mm   = hole_width_px  * mm_per_px
        height_mm  = hole_height_px * mm_per_px
        click.echo(f"  Hole size: {width_mm:.1f} × {height_mm:.1f} mm")

        # ── 7. Build mm-space outline ─────────────────────────────────────────
        outline_mm = (pts_smooth - np.array([x_min, y_min])) * mm_per_px
        outline_mm = np.vstack([outline_mm, outline_mm[0]])  # close polygon

        shape = MissingPieceShape(
            outline=outline_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            pixel_to_mm_scale=mm_per_px,
        )

        # ── 8. Debug images ───────────────────────────────────────────────────
        ddir = Path(debug_dir) if debug_dir else Path(output).parent

        # Detection debug: overlay hole contour on the original image
        det_img = image.copy()
        contour_pts = pts_smooth.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(det_img, [contour_pts], isClosed=True,
                      color=(0, 255, 0), thickness=3)
        cv2.circle(det_img, (seed_x, seed_y), 12, (0, 0, 255), -1)
        det_path = ddir / "debug_detection.jpg"
        ddir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(det_path), det_img)
        click.echo(f"  Debug detection image: {det_path}")

        # Shape debug: 2D outline rendering
        shape_img_path = ddir / "debug_shape.jpg"
        _save_shape_image(shape, shape_img_path)
        click.echo(f"  Debug shape image:     {shape_img_path}")

        # ── 9. Generate 3D model ──────────────────────────────────────────────
        click.echo(f"Generating 3D model (format={output_format})...")
        model_gen.generate(
            shape,
            output_path=output,
            format=output_format,
            thickness_mm=thickness,
            bevel_mm=bevel,
        )
        click.echo(f"Done. Output written to: {output}")

    except PipelineError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def _save_shape_image(shape: MissingPieceShape, output_path: Path) -> None:
    """Render the 2D outline as a JPEG debug image."""
    canvas = 900
    margin = 80

    outline = shape.outline
    x_vals, y_vals = outline[:, 0], outline[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    w_mm = x_max - x_min
    h_mm = y_max - y_min

    scale = (canvas - 2 * margin) / max(w_mm, h_mm, 0.001)

    img = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255
    pts = np.column_stack([
        (x_vals - x_min) * scale + margin,
        (y_vals - y_min) * scale + margin,
    ]).astype(np.int32)

    cv2.fillPoly(img, [pts], (230, 210, 180))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    label = f"{shape.width_mm:.1f} mm  x  {shape.height_mm:.1f} mm"
    cv2.putText(img, label, (margin, canvas - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

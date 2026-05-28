"""CLI entry point for missing-piece-gen."""

import sys
import os

import click

from missing_piece_gen import __version__
from missing_piece_gen.pipeline import (
    detect_missing_region,
    generate_3d_model,
    load_image,
    write_output,
)

OUTPUT_FORMATS = ("stl", "obj")


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
def main(image_path: str, output: str, output_format: str) -> None:
    """Generate a 3D-printable missing puzzle piece from IMAGE_PATH.

    The pipeline detects the missing region in IMAGE_PATH, builds a 3D
    model of the piece, and writes the result to the output file.
    """
    # Validate input path
    if not os.path.exists(image_path):
        click.echo(
            f"Error: image path does not exist: {image_path}", err=True
        )
        sys.exit(1)

    click.echo(f"Loading image: {image_path}")
    image_data = load_image(image_path)

    click.echo("Detecting missing region...")
    region_data = detect_missing_region(image_data)

    click.echo(f"Generating 3D model (format={output_format})...")
    model_data = generate_3d_model(region_data, output_format)

    click.echo(f"Writing output to: {output}")
    write_output(model_data, output)

    click.echo("Done.")

"""Pipeline stages for missing piece generation.

Each function is a stub that will be replaced with a real implementation
in later sprints. They return placeholder data suitable for wiring the
CLI end-to-end.
"""


def load_image(image_path: str) -> dict:
    """Load and pre-process the puzzle image.

    Args:
        image_path: Path to the source image file.

    Returns:
        A dict representing the loaded image data (stub).
    """
    return {"image_path": image_path, "width": 0, "height": 0}


def detect_missing_region(image_data: dict) -> dict:
    """Detect the region of the missing puzzle piece.

    Args:
        image_data: Output from load_image().

    Returns:
        A dict describing the missing region boundary (stub).
    """
    return {"region": None, "bounding_box": None}


def generate_3d_model(region_data: dict, output_format: str) -> dict:
    """Generate a 3D model for the missing piece.

    Args:
        region_data: Output from detect_missing_region().
        output_format: Either "stl" or "obj".

    Returns:
        A dict containing the model data (stub).
    """
    return {"format": output_format, "vertices": [], "faces": []}


def write_output(model_data: dict, output_path: str) -> None:
    """Write the 3D model to disk.

    Args:
        model_data: Output from generate_3d_model().
        output_path: Destination file path.
    """
    with open(output_path, "w") as fh:
        fh.write("# missing-piece-gen stub output\n")
        fh.write(f"# format: {model_data['format']}\n")

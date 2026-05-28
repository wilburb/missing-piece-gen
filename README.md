# missing-piece-gen

Generate a 3D-printable missing puzzle piece from an image.

## Requirements

- Python 3.9+

## Install

```bash
pip install -e .
```

This registers the `missing-piece-gen` command on your PATH.

## Usage

```
missing-piece-gen [OPTIONS] <image_path>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<image_path>` | Path to the puzzle image (JPEG, PNG, etc.) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output <path>`, `-o <path>` | `missing_piece.stl` | Output file path |
| `--format stl\|obj`, `-f stl\|obj` | `stl` | Output 3D model format |
| `--version` | | Show the version and exit |
| `--help` | | Show help and exit |

### Examples

Generate a missing piece using defaults:

```bash
missing-piece-gen puzzle.jpg
```

Write the result to a custom path in OBJ format:

```bash
missing-piece-gen puzzle.jpg --output piece.obj --format obj
```

## Development

Install with dev dependencies and run tests:

```bash
pip install -e ".[dev]"
pytest
```

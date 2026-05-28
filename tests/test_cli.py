"""Tests for the missing-piece-gen CLI (issue #3 AC coverage)."""

import os

import pytest
from click.testing import CliRunner

from missing_piece_gen.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def tmp_image(tmp_path):
    """A temporary file that stands in for a real image."""
    img = tmp_path / "puzzle.png"
    img.write_bytes(b"\x89PNG\r\n")  # minimal PNG-like stub
    return str(img)


# ---------------------------------------------------------------------------
# AC: --help exits 0 and prints usage
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_prints_usage(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "Usage:" in result.output

    def test_help_describes_image_path_argument(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "image_path" in result.output.lower() or "<image_path>" in result.output

    def test_help_describes_output_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--output" in result.output

    def test_help_describes_format_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "--format" in result.output


# ---------------------------------------------------------------------------
# AC: valid image path runs without crashing and prints a status message
# ---------------------------------------------------------------------------

class TestValidImagePath:
    def test_valid_path_exits_zero(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.stl")
        result = runner.invoke(main, [tmp_image, "--output", out])
        assert result.exit_code == 0, result.output

    def test_valid_path_prints_status(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.stl")
        result = runner.invoke(main, [tmp_image, "--output", out])
        assert "Done." in result.output

    def test_valid_path_prints_loading_message(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.stl")
        result = runner.invoke(main, [tmp_image, "--output", out])
        assert "Loading image" in result.output


# ---------------------------------------------------------------------------
# AC: invalid/missing path exits non-zero with error message
# ---------------------------------------------------------------------------

class TestInvalidImagePath:
    def test_missing_path_exits_nonzero(self, runner, tmp_path):
        result = runner.invoke(main, ["/nonexistent/path/image.png"])
        assert result.exit_code != 0

    def test_missing_path_prints_error(self):
        # Use a fresh runner with mix_stderr=True so stderr appears in output
        mixing_runner = CliRunner(mix_stderr=True)
        result = mixing_runner.invoke(main, ["/nonexistent/path/image.png"])
        assert "Error" in result.output or "does not exist" in result.output


# ---------------------------------------------------------------------------
# AC: --output flag is accepted
# ---------------------------------------------------------------------------

class TestOutputFlag:
    def test_output_flag_accepted(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "custom_output.stl")
        result = runner.invoke(main, [tmp_image, "--output", out])
        assert result.exit_code == 0, result.output

    def test_output_file_is_created(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "custom_output.stl")
        runner.invoke(main, [tmp_image, "--output", out])
        assert os.path.exists(out)

    def test_default_output_name(self, runner, tmp_image):
        """Without --output the default filename is used."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, [tmp_image])
            assert result.exit_code == 0, result.output
            assert os.path.exists("missing_piece.stl")

    def test_short_flag_o_accepted(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "short.stl")
        result = runner.invoke(main, [tmp_image, "-o", out])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# AC: --format stl|obj flag is accepted
# ---------------------------------------------------------------------------

class TestFormatFlag:
    def test_format_stl_accepted(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.stl")
        result = runner.invoke(main, [tmp_image, "--output", out, "--format", "stl"])
        assert result.exit_code == 0, result.output

    def test_format_obj_accepted(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.obj")
        result = runner.invoke(main, [tmp_image, "--output", out, "--format", "obj"])
        assert result.exit_code == 0, result.output

    def test_invalid_format_rejected(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.xyz")
        result = runner.invoke(main, [tmp_image, "--output", out, "--format", "xyz"])
        assert result.exit_code != 0

    def test_short_flag_f_accepted(self, runner, tmp_image, tmp_path):
        out = str(tmp_path / "out.obj")
        result = runner.invoke(main, [tmp_image, "-f", "obj", "--output", out])
        assert result.exit_code == 0, result.output

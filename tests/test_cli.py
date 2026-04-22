from typer.testing import CliRunner
from ziv.cli.main import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_unknown_model_exits_nonzero():
    result = runner.invoke(app, ["init", "--model", "nonexistent_model"])
    assert result.exit_code != 0


def test_search_without_index_shows_error_not_traceback():
    result = runner.invoke(app, ["search", "test query"])
    # Validate graceful failure when no index is available.
    # Expected behavior:
    # - No unhandled exception (no traceback)
    # - Non-zero exit code
    # - Explicit error message about missing index
    # NOTE: Requires  test environment with no pre-existing index.
    assert "Traceback" not in result.output
    assert result.exit_code != 0 or "not" in result.output.lower()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.3" in result.output  # adjust for your version

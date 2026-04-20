"""Download and verify the Ziv embedding model."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import huggingface_hub
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()

huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

REPO_ID_FAST = "ziv-ai/embedder-fast-onnx"

ZIV_HOME = Path.home() / ".ziv"
MODEL_DIR_FAST = ZIV_HOME / "models" / "embedder-fast-onnx"

REQUIRED_FILES = [
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "1_Pooling/config.json",
]


def _is_model_installed(model_dir: str | Path = MODEL_DIR_FAST) -> bool:
    """Return True if all required model files exist and are non-empty."""
    model_path = Path(model_dir)
    return all(
        (model_path / relative_path).is_file()
        and (model_path / relative_path).stat().st_size > 0
        for relative_path in REQUIRED_FILES
    )


def _cleanup(model_dir: str | Path) -> None:
    """Remove a partial model directory."""
    model_path = Path(model_dir)
    if model_path.is_dir():
        shutil.rmtree(model_path)


def download_model(
    model_dir: str | Path = MODEL_DIR_FAST,
    repo_id: str = REPO_ID_FAST,
) -> None:
    """Download the ONNX model from Hugging Face and verify required files."""
    model_path = Path(model_dir)

    if _is_model_installed(model_path):
        console.print(
            Panel(
                Text.assemble(
                    ("✔ Model already installed\n", "bold green"),
                    (str(model_path.resolve()), "dim"),
                ),
                border_style="green",
                title="[bold green]ziv[/bold green]",
                expand=False,
            )
        )
        return

    console.print(
        Panel(
            Text.assemble(
                ("Downloading model: ", "bold white"),
                (repo_id, "bold cyan"),
                ("\nDestination: ", "bold white"),
                (str(model_path.resolve()), "dim"),
            ),
            border_style="cyan",
            title="[bold cyan]ziv · model install[/bold cyan]",
            expand=False,
        )
    )

    try:
        with console.status(
            "[cyan]Downloading model files...[/cyan]",
            spinner="dots",
        ):
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(model_path),
            )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Download interrupted. Cleaning up...[/bold yellow]")
        _cleanup(model_path)
        raise
    except Exception as exc:
        _cleanup(model_path)
        raise RuntimeError(f"Model download failed: {exc}") from exc

    if not _is_model_installed(model_path):
        _cleanup(model_path)
        raise RuntimeError("Model verification failed: required files are missing.")

    console.print(
        Panel(
            Text.assemble(
                ("✔ Model installed successfully\n", "bold green"),
                (str(model_path.resolve()), "dim"),
            ),
            border_style="green",
            title="[bold green]ziv[/bold green]",
            expand=False,
        )
    )

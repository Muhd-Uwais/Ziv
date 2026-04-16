"""
downloader.py — Model downloader for ziv.
Downloads ONNX model files from Hugging Face and stores them locally.
"""

import os
import shutil
import signal
import sys
import logging

import huggingface_hub
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()
huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


# ── Constants ────────────────────────────────────────────────────────────────


REPO_ID_FAST = "ziv-ai/embedder-fast-onnx"

ZIV_HOME = os.path.join(os.path.expanduser("~"), ".ziv")
MODEL_DIR_FAST = os.path.join(ZIV_HOME, "models", "embedder-fast-onnx")

# All files required for the tool to work. Missing any = silent breakage.
REQUIRED_FILES = [
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    os.path.join("1_Pooling", "config.json"),
]

HF_RESOLVE_URL_FAST = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
HF_API_URL_FAST = "https://huggingface.co/api/models/{repo_id}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_model_installed(model_dir: str = MODEL_DIR_FAST, i=0) -> bool:
    """Return True only if all required files exist and are non-empty."""
    return all(
        os.path.isfile(os.path.join(model_dir, f))
        and os.path.getsize(os.path.join(model_dir, f)) > 0
        for f in REQUIRED_FILES
    )


def _cleanup(model_dir: str) -> None:
    """Delete the model directory — called on interrupted or failed downloads."""
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)


# ── Public API ───────────────────────────────────────────────────────────────


def download_model(model_dir: str = MODEL_DIR_FAST, repo_id: str = REPO_ID_FAST) -> None:
    """
    Download the ziv ONNX model from HuggingFace to model_dir.

    - Skips silently if the model is already fully installed.
    - Registers SIGINT/SIGTERM handlers to clean up on interruption.
    - Verifies required files after download; cleans up on failure.
    """

    # ── Already installed ────────────────────────────────────────────────────
    if _is_model_installed(model_dir):
        console.print(
            Panel(
                Text.assemble(
                    ("✔  Model already installed\n", "bold green"),
                    (f"   {os.path.abspath(model_dir)}", "dim"),
                ),
                border_style="green",
                title="[bold green]ziv[/bold green]",
                expand=False,
            )
        )
        return

    # ── Announce ─────────────────────────────────────────────────────────────
    console.print(
        Panel(
            Text.assemble(
                ("Downloading model: ", "bold white"),
                (repo_id, "bold cyan"),
                ("\nDestination:       ", "bold white"),
                (os.path.abspath(model_dir), "dim"),
            ),
            border_style="cyan",
            title="[bold cyan]ziv · model install[/bold cyan]",
            expand=False,
        )
    )

    # ── Fetch file list ──────────────────────────────────────────────────────
    with console.status("[cyan]Fetching file list from HuggingFace…[/cyan]"):
        pass

    # ── Interrupt handler — clean up partial download on Ctrl+C ──────────────
    def _on_interrupt(sig, frame):
        console.print(
            "\n[bold yellow]⚠  Download interrupted — cleaning up…[/bold yellow]")
        _cleanup(model_dir)
        console.print(
            "[bold red]✖  Partial files removed. Run again to retry.[/bold red]")
        sys.exit(1)

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    # ── Download ─────────────────────────────────────────────────────────────
    try:
        with console.status(
            "[cyan]Downloading model files — this may take a minute…[/cyan]",
            spinner="dots"
        ):
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=model_dir,
            )

    except Exception as exc:
        console.print(f"\n[bold red]✖  Download failed:[/bold red] {exc}")
        console.print("[yellow]  Cleaning up partial files…[/yellow]")
        _cleanup(model_dir)
        sys.exit(1)

    # ── Post-download verification ────────────────────────────────────────────
    if not _is_model_installed(model_dir, i=1):
        console.print(
            "[bold red]✖  Verification failed — required files missing or empty.[/bold red]")
        console.print("[yellow]  Cleaning up…[/yellow]")
        _cleanup(model_dir)
        sys.exit(1)

    console.print(
        Panel(
            Text.assemble(
                ("✔  Model installed successfully\n", "bold green"),
                (f"{os.path.abspath(model_dir)}", "dim"),
            ),
            border_style="green",
            title="[bold green]ziv[/bold green]",
            expand=False,
        )
    )

"""Index building pipeline for ziv — file loading, chunking, caching, and FAISS."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.table import Table

from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunk_directory
from ..core.vector_store import build_and_save
from ..api.process_manager import get_server_url
from .retriever import ServerUnavailable

logger = logging.getLogger(__name__)
console = Console()


def _make_progress() -> Progress:
    """Return a consistent progress bar with spinner and elapsed/remaining time."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=35, style="cyan", complete_style="bold green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=False,
    )


class BuildIndex:
    """Orchestrates indexing: load → chunk → cache → embed → FAISS."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.api_link = get_server_url()

    def __is_server_ready(self) -> bool:
        """Return True if the embedding server responds with 200."""
        if self.api_link is None:
            return False
        try:
            response = requests.get(self.api_link + "health", timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _load_manifest(self, manifest_path: str) -> dict[str, Any]:
        """Load cache manifest or return a default one."""
        if not os.path.exists(manifest_path):
            return {"dtype": "float32", "dim": self.dim, "items": {}}
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_embeddings(self, embeddings_path: str) -> np.ndarray:
        """Load existing embeddings or return an empty array."""
        if not os.path.exists(embeddings_path):
            return np.empty((0, self.dim), dtype=np.float32)
        return np.load(embeddings_path)

    def _save_manifest(self, manifest_path: str, manifest: dict[str, Any]) -> None:
        """Persist the manifest to disk."""
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

    def _save_embeddings(self, embeddings_path: str, embeddings: np.ndarray) -> None:
        """Persist embeddings to disk."""
        np.save(embeddings_path, embeddings.astype(np.float32))

    def _batched(self, items: list[dict[str, Any]], batch_size: int):
        """Yield batches of chunk content strings."""
        for i in range(0, len(items), batch_size):
            yield [item["content"] for item in items[i: i + batch_size]]

    def _embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int,
        progress: Progress | None = None,
        task_id: Any | None = None,
    ) -> np.ndarray | None:
        """Embed a list of chunks and return a float32 matrix or None on error."""
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)

        if not self.__is_server_ready():
            console.print(
                "[bold red]❌ Embedding server not reachable.[/bold red]")
            logger.error("index build failed: embedding server unreachable")
            raise ServerUnavailable("Embedding server is not running")

        url = f"{self.api_link}encode-chunks"

        output = np.empty((len(chunks), self.dim), dtype=np.float32)
        idx = 0

        try:
            for batch in self._batched(chunks, batch_size):
                response = requests.post(
                    url,
                    json={"chunks": batch},
                    timeout=(10.0, 120.0),  # connect + read
                )
                response.raise_for_status()
                batch_embeddings = np.asarray(
                    response.json(), dtype=np.float32)

                n = len(batch_embeddings)
                output[idx: idx + n] = batch_embeddings
                idx += n

                if progress is not None and task_id is not None:
                    progress.advance(task_id, 1)

            return output

        except Exception as e:
            logger.exception("Unexpected error during embedding generation")
            console.print(
                f"[bold red]❌ Failed to generate embeddings:[/bold red] {e}")
            return None

    def build_index(
        self,
        root_path: str,
        output_dir: str = ".ziv",
        extensions: set[str] | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Build the semantic index of a codebase.

        Loads files, chunks them, caches computed embeddings, and builds a FAISS index.
        """
        if extensions is None:
            extensions = {".py"}

        start_time = time.monotonic()
        absolute_path = os.path.realpath(root_path)

        console.print(
            Panel(
                f"[bold cyan]ziv[/bold cyan]  ·  Building codebase index\n[dim]{absolute_path}[/dim]",
                border_style="cyan",
                padding=(0, 2),
                expand=False,
            )
        )

        try:
            with _make_progress() as progress:
                # ── Scan files ────────────────────────────────────────────────────────
                t = progress.add_task("[cyan]Scanning files...", total=None)
                logger.info("Scanning directory: %s", root_path)
                files = load_files_from_directory(root_path, extensions)
                progress.update(
                    t,
                    description=f"[green]✓ Found {len(files)} files[/green]",
                    total=1,
                    completed=1,
                )

                # ── Chunk files ───────────────────────────────────────────────────────
                t = progress.add_task("[cyan]Chunking files...", total=None)
                logger.info(
                    "Splitting %d files into manageable chunks", len(files))
                all_chunks = chunk_directory(files)
                progress.update(
                    t,
                    description=f"[green]✓ Created {len(all_chunks)} chunks[/green]",
                    total=1,
                    completed=1,
                )

                if not all_chunks:
                    console.print("[bold green]Nothing to build!")
                    return

                # ── Cache setup ───────────────────────────────────────────────────────
                t = progress.add_task("[cyan]Checking cache...", total=None)
                cache_dir = os.path.join(output_dir, "cache")
                os.makedirs(cache_dir, exist_ok=True)

                manifest_path = os.path.join(cache_dir, "cache_manifest.json")
                embeddings_path = os.path.join(cache_dir, "embeddings.npy")

                manifest = self._load_manifest(manifest_path)
                embeddings = self._load_embeddings(embeddings_path)

                new_chunks = [c for c in all_chunks if c["id"]
                              not in manifest["items"]]
                cached_count = len(all_chunks) - len(new_chunks)
                progress.update(
                    t,
                    description=(
                        f"[green]✓ Cache  [dim]{cached_count} hit · {len(new_chunks)} new[/dim]"
                    ),
                    total=1,
                    completed=1,
                )
                logger.info("Calling API to generate embeddings")

                # ── Embed new chunks ──────────────────────────────────────────────────
                if new_chunks:
                    num_batches = math.ceil(len(new_chunks) / batch_size)
                    embed_task = progress.add_task(
                        f"[cyan]Embedding [bold white]{len(new_chunks)}[/bold white] new chunks...",
                        total=num_batches,
                    )

                    try:
                        new_embeddings = self._embed_chunks(
                            new_chunks, batch_size, progress, embed_task
                        )
                    except ServerUnavailable:
                        raise

                    if new_embeddings is None:
                        console.print(
                            "[bold red]❌ Embedding failed. Aborting.[/bold red]")
                        return

                    progress.update(
                        embed_task,
                        description=f"[green]✓ Embedded {len(new_chunks)} chunks[/green]",
                    )

                    start_row = len(embeddings) if embeddings.size > 0 else 0
                    for i, chunk in enumerate(new_chunks):
                        manifest["items"][chunk["id"]] = {"row": start_row + i}

                    if embeddings.size == 0:
                        embeddings = new_embeddings
                    else:
                        embeddings = np.concatenate(
                            [embeddings, new_embeddings], axis=0, dtype=np.float32
                        )

                    self._save_manifest(manifest_path, manifest)
                    self._save_embeddings(embeddings_path, embeddings)

                else:
                    t = progress.add_task(
                        "[dim]All chunks cached — skipping embedding[/dim]", total=1
                    )
                    progress.advance(t)

                # ── Build FAISS index ─────────────────────────────────────────────────
                t = progress.add_task(
                    "[cyan]Building FAISS index...", total=None)

                valid_indices: list[int] = []
                metadata: list[dict[str, Any]] = []

                for chunk in all_chunks:
                    item = manifest["items"].get(chunk["id"])
                    if item is None:
                        logger.warning(
                            "Missing embedding for chunk %s, skipping", chunk["id"])
                        continue

                    valid_indices.append(item["row"])
                    metadata.append(
                        {
                            "id": chunk["id"],
                            "file_path": chunk["file_path"],
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "content": chunk["content"],
                            "embedding_index": len(valid_indices) - 1,
                        }
                    )

                if not valid_indices:
                    console.print(
                        Panel(
                            "[yellow]No valid chunks to index (possibly corrupted cache).[/yellow]",
                            border_style="yellow",
                            title="[bold]ziv[/bold]",
                            expand=False,
                        )
                    )
                    return

                final_embeddings = embeddings[valid_indices].astype(
                    np.float32, copy=False)
                build_and_save(final_embeddings, metadata, output_dir)

                progress.update(
                    t,
                    description=f"[green]✓ FAISS index ready[/green]  [dim]({len(valid_indices)} vectors)[/dim]",
                    total=1,
                    completed=1,
                )

                # ── Final summary ─────────────────────────────────────────────────────
                elapsed = time.monotonic() - start_time

                summary = Table(show_header=False, box=None, padding=(0, 2))
                summary.add_column(style="dim")
                summary.add_column(style="bold white")
                summary.add_row("Total chunks", str(len(all_chunks)))
                summary.add_row("Cached  (reused)", str(cached_count))
                summary.add_row("Newly embedded", str(len(new_chunks)))
                summary.add_row("Index saved to", str(output_dir))
                summary.add_row("Time elapsed", f"{elapsed:.1f}s")

                console.print(
                    Panel(
                        summary,
                        title="[bold green]✅ Index built successfully[/bold green]",
                        border_style="green",
                        padding=(0, 2),
                        expand=False,
                    )
                )

                logger.info(
                    "Index build completed — %d total, %d new, %d cached, %.1fs",
                    len(all_chunks),
                    len(new_chunks),
                    cached_count,
                    elapsed,
                )

        except ServerUnavailable:
            console.clear()
            console.print("[red]❌ Embedding server not reachable.[/red]")
            console.print(
                "[white]Start server with [cyan]`ziv start`[/cyan] first.[/white]")
            return

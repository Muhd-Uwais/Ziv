import os
import json
import requests
import logging
import numpy as np
import time
import math

from rich.console import Console
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
from rich.panel import Panel
from rich.table import Table

from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunk_directory
from ..core.vector_store import build_and_save
from ..api.process_manager import get_server_url


logger = logging.getLogger(__name__)
console = Console()


def _make_progress() -> Progress:
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
    def __init__(self, dim=384):
        self.api_link = get_server_url()
        self.dim = dim

    def __is_server_ready(self):
        if self.api_link is None:
            raise RuntimeError(
                "Embedding server is not running. Start it with `ziv start`.")
        try:
            response = requests.get(self.api_link+"health", timeout=1)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _load_manifest(self, manifest_path):
        if not os.path.exists(manifest_path):
            return {
                "dtype": "float32",
                "dim": self.dim,
                "items": {}
            }
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embeddings(self, embeddings_path):
        if not os.path.exists(embeddings_path):
            return np.empty((0, self.dim), dtype=np.float32)
        return np.load(embeddings_path)

    def _save_manifest(self, manifest_path, manifest):
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

    def _save_embeddings(self, embeddings_path, embeddings):
        np.save(embeddings_path, embeddings.astype(np.float32))

    def _batched(self, items, batch_size):
        for i in range(0, len(items), batch_size):
            yield [item["content"] for item in items[i:i + batch_size]]

    def _embed_chunks(
        self,
        chunks,
        batch_size,
        progress: Progress = None,
        task_id=None
    ) -> np.ndarray:
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)

        if not self.__is_server_ready():
            raise RuntimeError("Embedding server is not reachable")

        url = self.api_link + "encode-chunks"

        output = np.empty((len(chunks), self.dim), dtype=np.float32)
        idx = 0

        try:
            for batch in self._batched(chunks, batch_size):
                response = requests.post(url, json=batch, timeout=120)
                response.raise_for_status()
                batch_embeddings = np.asarray(
                    response.json(), dtype=np.float32)

                n = len(batch_embeddings)
                output[idx:idx + n] = batch_embeddings
                idx += n

                if progress is not None and task_id is not None:
                    # advance by 1 batch, not 1 chunk
                    progress.advance(task_id)

            return output
        except Exception as e:
            logger.exception(
                "Unexpected error during embeddingg generation")
            console.print(
                f"\n[bold red]❌ Failed to generate embeddings:[/bold red] {e}")
            return

    def build_index(self, root_path, output_dir=".ziv", extensions={".py"}, batch_size=32):

        start_time = time.monotonic()

        console.print(Panel(
            f"[bold cyan]ziv[/bold cyan]  ·  Building codebase index\n"
            f"[dim]{os.path.realpath(root_path)}[/dim]",
            border_style="cyan",
            padding=(0, 2),
            expand=False
        ))

        with _make_progress() as progress:

            t = progress.add_task("[cyan]Scanning files...", total=None)
            logger.info(f"Scanning directory: {root_path}")
            files = load_files_from_directory(root_path, extensions)
            progress.update(
                t,
                description=f"[green]✓ Found {len(files)} files[/green]",
                total=1, completed=1,
            )

            t = progress.add_task("[cyan]Chunking files...", total=None)
            logger.info(f"Splitting {len(files)} files into manageble chunks")
            all_chunks = chunk_directory(files)
            progress.update(
                t,
                description=f"[green]✓ Created {len(all_chunks)} chunks[/green]",
                total=1, completed=1,
            )

            if not all_chunks:
                console.print("[bold green]Nothing to built!")
                return

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
                    f"[green]✓ Cache[/green]  "
                    f"[dim]{cached_count} hit · {len(new_chunks)} new[/dim]"
                ),
                total=1, completed=1,
            )
            logger.info("Calling api to generate embeddings")

            # new_embeddings = self._embed_chunks(new_chunks, batch_size)

            if new_chunks:
                num_batches = math.ceil(len(new_chunks) / batch_size)
                embed_task = progress.add_task(
                    f"[cyan]Embedding [bold white]{len(new_chunks)}[/bold white] new chunks...",
                    total=num_batches,
                )

                new_embeddings = self._embed_chunks(
                    new_chunks, batch_size, progress, embed_task
                )

                if new_embeddings is None:
                    console.print(
                        "[bold red]❌  Embedding failed. Aborting.[/bold red]")
                    return

                progress.update(
                    embed_task,
                    description=f"[green]✓ Embedded {len(new_chunks)} chunks[/green]",
                )

                start_row = len(embeddings) if embeddings.size > 0 else 0
                for i, chunk in enumerate(new_chunks):
                    manifest["items"][chunk["id"]] = {"row": start_row + i}

                embeddings = (
                    new_embeddings
                    if embeddings.size == 0
                    else np.concatenate([embeddings, new_embeddings], axis=0)
                )

                self._save_manifest(manifest_path, manifest)
                self._save_embeddings(embeddings_path, embeddings)
            else:
                t = progress.add_task(
                    "[dim]All chunks cached — skipping embedding[/dim]",
                    total=1,
                )
                progress.advance(t)

            t = progress.add_task("[cyan]Building FAISS index...", total=None)

            valid_indices = []
            metadata = []
            for chunk in all_chunks:
                item = manifest["items"].get(chunk["id"])
                if item is None:
                    logger.warning(
                        "Missing embedding for chunk %s, skipping", chunk["id"])
                    continue

                valid_indices.append(item["row"])
                metadata.append(
                    {
                        "id": chunk['id'],
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["content"],
                        "embedding_index": len(valid_indices) - 1,
                    }
                )

            final_embeddings = embeddings[valid_indices].astype(
                np.float32, copy=False)
            build_and_save(final_embeddings, metadata, output_dir)

            progress.update(
                t,
                description=f"[green]✓ FAISS index ready[/green]  [dim]({len(valid_indices)} vectors)[/dim]",
                total=1, completed=1,
            )

        elapsed = time.monotonic() - start_time

        summary = Table(show_header=False, box=None, padding=(0, 2))
        summary.add_column(style="dim")
        summary.add_column(style="bold white")
        summary.add_row("Total chunks",     str(len(all_chunks)))
        summary.add_row("Cached  (reused)", str(cached_count))
        summary.add_row("Newly embedded",   str(len(new_chunks)))
        summary.add_row("Index saved to",   output_dir)
        summary.add_row("Time elapsed",     f"{elapsed:.1f}s")

        console.print(Panel(
            summary,
            title="[bold green]✅  Index built successfully[/bold green]",
            border_style="green",
            padding=(0, 2),
            expand=False
        ))

        logger.info(
            "Index build completed — %d total, %d new, %d cached, %.1fs",
            len(all_chunks), len(new_chunks), cached_count, elapsed,
        )

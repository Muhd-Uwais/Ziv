import os
import json
import requests
import logging
from rich.console import Console
from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunk_directory
from ..core.vector_store import build_and_save


logger = logging.getLogger(__name__)
console = Console()


class BuildIndex:
    def __init__(self, api_link="http://localhost:8000/"):
        self.api_link = api_link

    def __is_server_ready(self):
        try:
            response = requests.get(self.api_link+"health", timeout=1)
            return response.status_code == 200
        except:
            return False

    def load_cache(self, cache_path):
        if not os.path.exists(cache_path):
            return {}

        with open(cache_path, "r") as f:
            return json.load(f)

    def build_index(self, root_path, output_dir=".ziv", extensions={".py"}):
        with console.status("[bold green]Building your codebase index...", spinner="dots") as status:
            status.update("[bold blue]Loading files...")
            logger.info(f"Scanning directory: {root_path}")
            files = load_files_from_directory(root_path, extensions)

            status.update("[bold blue]Chunking files...")
            logger.info(f"Splitting {len(files)} files into manageble chunks")
            all_chunks = chunk_directory(files)

            if not all_chunks:
                console.print("[bold green]Nothing to built!")
                return

            os.makedirs(os.path.join(output_dir, "cache"), exist_ok=True)
            chunk_cache_path = os.path.join(
                output_dir, "cache", "chunk_cache.json")
            chunk_cache = self.load_cache(chunk_cache_path)

            new_chunks = []
            cached_embeddings = {}  # chunk_id → embedding

            for chunk in all_chunks:
                if chunk["id"] in chunk_cache:
                    cached_embeddings[chunk["id"]] = chunk_cache[chunk["id"]]
                else:
                    new_chunks.append(chunk)

            status.update(
                f"[bold blue]Generating embeddings for {len(new_chunks)} chunks...")
            logger.info("Calling api to generate embeddings")
            url = self.api_link + "encode-chunks"

            if new_chunks:
                try:
                    if self.__is_server_ready():
                        response = requests.post(url, json=new_chunks)
                        response.raise_for_status()
                        new_embeddings = response.json()
                    else:
                        logger.error("Server not reachable at %s", url)
                        console.print(
                            "\n[bold red]❌ Error:[/bold red] Failed to connect to server. Is background embeddings server running?")
                        return
                except Exception as e:
                    status.stop()
                    logger.exception(
                        "Unexpected error during embedding generation")
                    console.print(
                        f"\n[bold red]❌ Failed to generate embeddings:[/bold red] {e}")
                    return

                # Save new embeddings into chunk cache
                for chunk, emb in zip(new_chunks, new_embeddings):
                    chunk_cache[chunk["id"]] = emb
            else:
                console.print(
                    "[dim]No new chunks to embed — using cache.[/dim]")

            final_embeddings = []
            metadata = []
            for chunk in all_chunks:
                if chunk["id"] not in chunk_cache:
                    logger.warning(
                        "Missing embedding for chunk %s, skipping", chunk["id"])
                    continue
                final_embeddings.append(chunk_cache[chunk["id"]])
                metadata.append(
                    {
                        "id": chunk['id'],
                        "file_path": chunk["file_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["content"],
                        "embedding_index": len(final_embeddings) - 1,
                    }
                )

            with open(chunk_cache_path, "w") as f:
                json.dump(chunk_cache, f)

            status.update("[bold blue]Building FAISS index...")
            build_and_save(final_embeddings, metadata, output_dir)

        console.print(
            f"✅ [bold green]Index built successfully![/bold green] Saved to [cyan]{output_dir}/[/cyan]")
        logger.info("Index build completed for %d chunks", len(all_chunks))

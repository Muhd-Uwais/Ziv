import os
import json
import requests
import logging
import numpy as np
from rich.console import Console
from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunks_directory


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

    def build_index(self, root_path, output_dir=".lfit", extensions={".py"}):
        with console.status("[bold green]Building your codebase index...", spinner="dots") as status:

            status.update("[bold blue]Loading files...")
            logger.info(f"Scanning directory: {root_path}")
            files = load_files_from_directory(root_path, extensions)

            status.update("[bold blue]Chunking files...")
            logger.info(f"Splitting {len(files)} files into manageble chunks")
            all_chunks = chunks_directory(files)

            status.update(
                f"[bold blue]Generating embeddings for {len(all_chunks)} chunks...")
            logger.info("Calling api to generate embeddings")
            url = self.api_link + "encode-chunks"

            embeddings = None
            try:
                if self.__is_server_ready():
                    response = requests.post(url, json=all_chunks)
                    response.raise_for_status()
                    embeddings = response.json()
                else:
                    logger.error("Server not reachable at %s", url)
                    console.print(
                        "\n[bold red]❌ Error:[/bold red] Failed to connect to server. Is background embeddings server running?")
                    return
            except Exception as e:
                logger.exception(
                    "Unexpected error during embedding generation")
                console.print(
                    f"\n[bold red]❌ Failed to generate embeddings:[/bold red] {e}")
                return

            status.update("[bold blue]Saving index to disk...")
            os.makedirs(output_dir, exist_ok=True)

            np.save(os.path.join(output_dir, "embeddings.npy"),
                    np.array(embeddings))

        # Save metadata
        metadata = [
            {
                "id": chunk["id"],
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["content"],
                "embedding_index": i,
            }
            for i, chunk in enumerate(all_chunks)
        ]

        with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        console.print(
            f"✅ [bold green]Index built successfully![/bold green] Saved to [cyan]{output_dir}/[/cyan]")
        logger.info("Index build completed for %d chunks", len(all_chunks))

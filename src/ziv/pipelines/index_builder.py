import os
import json
import requests
import logging
import numpy as np
from rich.console import Console
from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunk_directory
from ..core.vector_store import build_and_save


logger = logging.getLogger(__name__)
console = Console()


class BuildIndex:
    def __init__(self, api_link="http://localhost:8000/", dim=384):
        self.api_link = api_link
        self.dim = dim

    def __is_server_ready(self):
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

    def _embed_chunks(self, chunks, batch_size):
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
                batch_embeddings = np.asarray(response.json(), dtype=np.float32)
                
                n = len(batch_embeddings)
                output[idx:idx + n] = batch_embeddings
                idx += n
            return output  
        except Exception as e:
            logger.exception(
                        "Unexpected error during embeddingg generation")
            console.print(
                        f"\n[bold red]❌ Failed to generate embeddings:[/bold red] {e}")
            return
    
    def build_index(self, root_path, output_dir=".ziv", extensions={".py"}, batch_size=32):
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

            cache_dir = os.path.join(output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            manifest_path = os.path.join(cache_dir, "cache_manifest.json")
            embeddings_path = os.path.join(cache_dir, "embeddings.npy")

            manifest = self._load_manifest(manifest_path)
            embeddings = self._load_embeddings(embeddings_path)

            new_chunks = []
            for chunk in all_chunks:
                if chunk["id"] not in manifest["items"]:
                    new_chunks.append(chunk)

            status.update(
                f"[bold blue]Generating embeddings for {len(new_chunks)} chunks...")
            logger.info("Calling api to generate embeddings")

            new_embeddings = self._embed_chunks(new_chunks, batch_size)

            if new_chunks:
                start_row = len(embeddings) if embeddings.size > 0 else 0

                for i, chunk in enumerate(new_chunks):
                    manifest["items"][chunk["id"]] = {"row": start_row + i}

                if embeddings.size == 0:
                    embeddings = new_embeddings
                else:
                    embeddings = np.concatenate([embeddings, new_embeddings], axis=0)

                self._save_manifest(manifest_path, manifest)
                self._save_embeddings(embeddings_path, embeddings)
            else:
                console.print("[dim]No new chunks to embed[/dim]")                

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

            final_embeddings = embeddings[valid_indices].astype(np.float32, copy=False) 

            status.update("[bold blue]Building FAISS index...")
            build_and_save(final_embeddings, metadata, output_dir)

        console.print(
            f"✅ [bold green]Index built successfully![/bold green] Saved to [cyan]{output_dir}/[/cyan]")
        logger.info("Index build completed for %d chunks", len(all_chunks))

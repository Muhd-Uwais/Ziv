import os
import json
import requests
import logging
import numpy as np
from ..core.file_loader import load_files_from_directory
from ..core.chunker import chunks_directory


logger = logging.getLogger(__name__)


class BuildIndex:
    def __init__(self, api_link="http://localhost:8000/"):
        self.api_link = api_link

    def build_index(self, root_path, output_dir=".lfit", extensions={".py"}):
        logger.info("Loading files...")
        files = load_files_from_directory(root_path, extensions)

        logger.info("Chunking files...")
        all_chunks = chunks_directory(files)

        logger.info("Generating embeddings...")
        url = self.api_link + "encode-chunks"
        embeddings = (requests.post(url, json=all_chunks)).json()

        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/embeddings.npy", embeddings)

        # Save metadata
        metadata = []
        for i, chunk in enumerate(all_chunks):
            metadata.append({
                "id": chunk["id"],
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["content"],
                "embedding_index": i
            })

        with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Index build successfully...")
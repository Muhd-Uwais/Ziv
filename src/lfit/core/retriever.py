import numpy as np
import logging
import requests
import json
from rich.console import Console


logger = logging.getLogger(__name__)
console = Console()


class Retriever:
    def __init__(self, index_path=".lfit"):
        self.embeddings = np.load(f"{index_path}/embeddings.npy")
        self.api_link = "http://localhost:8000/"

        with open(f"{index_path}/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # sanity check
        assert len(self.embeddings) == len(self.metadata), \
            "Embeddings and metadata size mismatch"

    def __is_server_ready(self):
        try:
            response = requests.get(self.api_link+"health", timeout=1)
            return response.status_code == 200
        except:
            return False

    def search(self, query, top_k=5):
        """
        Perform similarity search.

        Args.
            query_embedding (np.ndarray)
            top_k (int)

        Returns:
            List[dict]    
        """

        # Brief status check for the API call
        with console.status("[bold green]Analyzing query...", spinner="point"):
            url = self.api_link + "encode-query"
            query_payload = {"query": query}
            query_embedding = None
            try:
                if self.__is_server_ready():
                    response = requests.post(url, json=query_payload)
                    response.raise_for_status()
                    query_embedding = np.array(response.json())
                    logger.info(f"Successfully encoded query: '{query}'")
                else:
                    logger.error(
                        f"Search failed: Server unreachable at {self.api_link+'health'}")
                    console.print(
                        "\n[bold red]❌ Error:[/bold red] Embedding Server is not running. Please start it first.")
                    return []
            except Exception as e:
                logger.exception("Query encoding failed")
                console.print(f"\n[bold red]❌ Search Error:[/bold red] {e}")
                return []

        # Compute similarity (dot product)
        try:
            scores = np.dot(self.embeddings, query_embedding)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            logger.info(
                f"Computed similarity scores for {len(scores)} chunks.")
        except Exception as e:
            logger.error(f"Computation error: {e}")
            console.print(
                "[bold red]❌ Error calculating similarity scores.[/bold red]")
            return []

        results = [
            {
                "score": float(scores[idx]),
                "file_path": self.metadata[idx]["file_path"],
                "start_line": self.metadata[idx]["start_line"],
                "end_line": self.metadata[idx]["end_line"],
                "content": self.metadata[idx]["content"]
            }
            for idx in top_indices
        ]

        return results

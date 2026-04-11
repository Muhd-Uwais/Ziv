import numpy as np
import logging
import requests
from rich.console import Console
from ..core.vector_store import load, is_index_built
from ..core import vector_store


logger = logging.getLogger(__name__)
console = Console()


class Retriever:
    def __init__(self, index_path=".ziv"):
        self.api_link = "http://localhost:8000/"

        if not is_index_built(index_path):
            console.print(
                "[bold red]❌ No index found.[/bold red] Run [cyan]ziv build-index[/cyan] first."
            )
            raise FileNotFoundError(f"No FAISS index in '{index_path}'")

        self.index, self.id_map = load(index_path)

    def __is_server_ready(self):
        try:
            response = requests.get(self.api_link+"health", timeout=1)
            return response.status_code == 200
        except:
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform similarity search.

        Args.
            query (np.ndarray)
            top_k (int)

        Returns:
            List[dict]
        """

        # Brief status check for the API call
        with console.status("[bold green]Analyzing query...", spinner="point"):
            if not self.__is_server_ready():
                logger.error("Search failed: server unreachable")
                console.print(
                    "\n[bold red]❌ Error:[/bold red] Embedding server is not running. "
                    "Start it with [cyan]ziv start[/cyan]."
                )
                return -1

            try:
                response = requests.post(
                    self.api_link + "encode-query",
                    json={"query": query}
                )
                response.raise_for_status()
                query_vector = np.array(response.json(), dtype=np.float32)
                logger.info(f"Successfully encoded query: '{query}'")
            except Exception as e:
                logger.exception("Query encoding failed")
                console.print(f"\n[bold red]❌ Search Error:[/bold red] {e}")
                return []

        # Compute similarity (dot product)
        try:
            results = vector_store.search(
                self.index,
                self.id_map,
                query_vector,
                k=top_k
            )

            logger.info(
                f"Computed similarity scores for {len(results)} chunks.")
        except Exception as e:
            logger.error(f"Computation error: {e}")
            console.print(
                "[bold red]❌ Error calculating similarity scores.[/bold red]")
            return []

        return results
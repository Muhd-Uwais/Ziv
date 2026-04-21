"""Retriever pipeline for semantic search over a built Ziv index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rich.console import Console

from ..api.process_manager import get_server_url
from ..core.vector_store import load, is_index_built, search as search_index


logger = logging.getLogger(__name__)
console = Console()


class ServerUnavailable(Exception): pass


class Retriever:
    """Load the persisted index and run semantic search queries against it."""

    def __init__(self, index_path: str | Path = ".ziv") -> None:
        self.index_path = Path(index_path)
        self.api_link = get_server_url()

        if not is_index_built(self.index_path):
            console.print(
                "[bold red]❌ No index found.[/bold red] "
                "Run [cyan]ziv build-index[/cyan] first."
            )
            raise FileNotFoundError(
                f"No FAISS index found in '{self.index_path}'")

        self.index, self.id_map = load(self.index_path)

    def __is_server_ready(self) -> bool:
        """Return True if the embedding server is reachable."""
        if self.api_link is None:
            return False

        try:
            response = requests.get(
                f"{self.api_link}health", timeout=(3.0, 5.0))
            return response.status_code == 200
        except requests.RequestException:
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]] | None:
        """
        Encode a natural-language query and return the most relevant chunks.

        Returns:
          A list of result dicts on success.
          None if the embedding server is unavailable.
        """
        with console.status("[bold green]Analyzing query...[/bold green]", spinner="point"):
            if not self.__is_server_ready():
                logger.error("Search failed: embedding server unreachable")
                raise ServerUnavailable("Embedding server is not running")

            try:
                response = requests.post(
                    f"{self.api_link}encode-query",
                    json={"query": query},
                    timeout=(5.0, 30.0),
                )
                response.raise_for_status()
                query_vector = np.asarray(response.json(), dtype=np.float32)
                logger.info("Successfully encoded query: %r", query)

            except requests.RequestException as exc:
                logger.exception("Query encoding request failed")
                console.print(f"[bold red]❌ Search error:[/bold red] {exc}")
                return []

            except ValueError as exc:
                logger.exception("Invalid query embedding response")
                console.print(
                    f"[bold red]❌ Invalid embedding response:[/bold red] {exc}")
                return []

        try:
            results = search_index(
                self.index,
                self.id_map,
                query_vector,
                k=top_k,
            )
            logger.info(
                "Computed similarity scores for %d chunks", len(results))
            return results

        except Exception as exc:
            logger.exception("Similarity search failed")
            console.print(
                "[bold red]❌ Error calculating similarity scores.[/bold red]")
            logger.error("Computation error: %s", exc)
            return []

import numpy as np
import logging
import requests
import json

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, index_path=".lfit"):
        self.embeddings = np.load(f"{index_path}/embeddings.npy")
        self.api_link = "http://localhost:8000/"

        with open(f"{index_path}/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # sanity check
        assert len(self.embeddings) ==  len(self.metadata), \
            "Embeddings and metadata size mismatch"    
        
    def search(self, query, top_k=5):
        """
        Perform similarity search.

        Args.
            query_embedding (np.ndarray)
            top_k (int)

        Returns:
            List[dict]    
        """    

        # Generate embeddings of query
        url = self.api_link + "encode-query"
        query_payload = {"query": query}
        query_embedding = requests.post(url, json=query_payload).json()

        # Compute similarity (dot product)
        scores = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append({
                "score": float(scores[idx]),
                "file_path": meta["file_path"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "content": meta["content"]
            })

        return results
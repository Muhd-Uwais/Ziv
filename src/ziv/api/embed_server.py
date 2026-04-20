"""FastAPI service for lightweight text embedding."""

from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from .embedder import LightEmbedder
from pydantic import BaseModel
from typing import Any, Optional
import os
import logging


logger = logging.getLogger(__name__)

MODEL_NAME = "embedder-fast-onnx"

ZIV_HOME = os.path.join(os.path.expanduser("~"), ".ziv")
MODEL_DIR = os.path.join(ZIV_HOME, "models", MODEL_NAME)


class ModelContainer:
    """Holds the shared embedding model instance."""

    def __init__(self):
        self.model: Optional[LightEmbedder] = None
        self.status = "initializing"

    def load(self):
        """Load the embedding model into memory once at startup."""        
        try:
            logger.info(f"Loading model from {MODEL_DIR}")
            self.model = LightEmbedder(MODEL_DIR)
            self.status = "Ready"
            logger.info("Model ready")
        except Exception as e:
            self.status = f"error: {str(e)}"
            raise e


# --- INITIALIZATION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources on startup and release them on shutdown."""
    container.load()
    yield
    container.model = None

container = ModelContainer()
app = FastAPI(lifespan=lifespan)


# --- SCHEMAS ---
class QueryModel(BaseModel):
    query: str


# --- ENDPOINTS ---
@app.post("/encode-chunks")
def encode_chunks(chunks: list[Any] = Body(...)) -> list[list[float]]:
    """Encode a batch of text chunks into embedding vectors."""

    if container.status != "Ready":
        raise HTTPException(503, detail=f"Model not ready: {container.status}")

    embeddings = container.model.encode(chunks)

    return embeddings.tolist()


@app.post("/encode-query")
def encode_query(request: QueryModel) -> list[float]:
    """Encode a single query into an embedding vector."""

    if container.status != "Ready":
        raise HTTPException(503, detail=f"Model not ready: {container.status}")

    if not request.query:
        raise HTTPException(422, detail="texts list cannot be empty")

    embeddings = container.model.encode(request.query)

    return embeddings.tolist()


@app.get("/health")
def health():
    """Return service and model health status."""
    return {
        "status": "ok",
        "model_status": container.status,
        "model_name": MODEL_NAME
    }

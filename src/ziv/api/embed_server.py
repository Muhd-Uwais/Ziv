"""FastAPI service for lightweight text embedding."""

from contextlib import asynccontextmanager
import logging
import os
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from .embedder import LightEmbedder


logger = logging.getLogger(__name__)

MODEL_NAME = "embedder-fast-onnx"
ZIV_HOME = os.path.join(os.path.expanduser("~"), ".ziv")
MODEL_DIR = os.path.join(ZIV_HOME, "models", MODEL_NAME)


class QueryRequest(BaseModel):
    """Request body for single-query embedding."""

    query: str = Field(..., min_length=1, description="Query text to encode.")


class ChunksRequest(BaseModel):
    """Request body for batch text embedding."""

    chunks: list[str] = Field(
        ...,
        min_length=1,
        description="Non-empty list of text chunks to encode."
    )

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    model_status: str
    model_name: str


class ModelContainer:
    """Holds the shared embedding model instance."""

    def __init__(self) -> None:
        self.model: LightEmbedder | None = None
        self.status: str = "initializing"

    def load(self) -> None:
        """Load the embedding model into memory once at startup."""
        logger.info("Loading model from %s", MODEL_DIR)
        self.model = LightEmbedder(MODEL_DIR)
        self.status = "Ready"
        logger.info("Model ready")

    def unload(self) -> None:
        """Release model reference during shutdown."""
        self.model = None
        self.status = "stopped"

    def get_model(self) -> LightEmbedder:
        """Return the loaded model or raise a 503 error."""
        if self.model is None or self.status != "Ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not ready: {self.status}"
            )
        return self.model


container = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources on startup and release them on shutdown."""
    try:
        container.load()
        yield
    except Exception as exc:
        container.status = f"error: {exc}"
        logger.exception("Failed to start embedding service")
        raise
    finally:
        container.unload()


app = FastAPI(
    title="Embedding Service",
    version="0.3.0",
    lifespan=lifespan,
)


@app.post("/encode-chunks")
def encode_chunks(
    request: Annotated[
        ChunksRequest,
        Body(description="Batch of text chunks to encode.")
    ]
) -> list[list[float]]:
    """Encode a batch of text chunks into embedding vectors."""
    model = container.get_model()
    embeddings = model.encode(request.chunks)
    return embeddings.tolist()


@app.post("/encode-query")
def encode_query(request: QueryRequest) -> list[float]:
    """Encode a single query into an embedding vector."""
    model = container.get_model()
    embeddings = model.encode(request.query)[0]
    return embeddings.tolist()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service and model health status."""
    return HealthResponse(
        status="ok",
        model_status=container.status,
        model_name=MODEL_NAME,
    )

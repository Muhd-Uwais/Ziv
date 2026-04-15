from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from .embedder import LightEmbeddder
from pydantic import BaseModel
from typing import Any, Optional
import os
import logging


logger = logging.getLogger(__name__)

MODEL_NAME = "embedder-fast-onnx"
MODEL_DIR = os.path.join(os.getcwd(), ".ziv/models", MODEL_NAME)


class ModelContainer:
    def __init__(self):
        self.model: Optional[LightEmbeddder] = None
        self.status = "initializing"

    def load(self):
        try:
            logger.info(f"Loading model from {MODEL_DIR}")
            self.model = LightEmbeddder(MODEL_DIR)
            self.status = "Ready"
            logger.info("Model ready")
        except Exception as e:
            self.status = f"error: {str(e)}"
            raise e


# --- INITIALIZATION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
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
def encode_chunks(chunks: list[Any] = Body(...)):

    if container.status != "Ready":
        raise HTTPException(503, detail=f"Model not ready: {container.status}")

    embeddings = container.model.encode(chunks)

    return embeddings.tolist()


@app.post("/encode-query")
def encode_query(request: QueryModel):

    if container.status != "Ready":
        raise HTTPException(503, detail=f"Model not ready: {container.status}")

    if not request.query:
        raise HTTPException(422, detail="texts list cannot be empty")

    embeddings = container.model.encode(request.query)

    return embeddings.tolist()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_status": container.status,
        "model_name": MODEL_NAME
    }

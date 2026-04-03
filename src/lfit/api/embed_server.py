from fastapi import FastAPI
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
import os


MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = os.path.join(os.getcwd(), ".lfit/models", MODEL_NAME)


class ModelContainer:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.status = "initializing"

    def load(self):
        try:
            if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
                self.model = SentenceTransformer(
                    MODEL_DIR, local_files_only=True)
            else:
                self.model = SentenceTransformer(MODEL_NAME)
                os.makedirs(MODEL_DIR, exist_ok=True)
                self.model.save(MODEL_DIR)
            self.status = "ready"
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
class Chunk(BaseModel):
    id: UUID
    file_path: str
    start_line: int
    end_line: int
    content: str


class QueryModel(BaseModel):
    query: str


# --- ENDPOINTS ---
@app.post("/encode-chunks")
def encode_chunks(chunks: List[Chunk], batch_size=32):

    texts = [chunk.content for chunk in chunks]

    embeddings = container.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT for cosine similarity
    )

    return embeddings.tolist()


@app.post("/encode-query")
def encode_query(data: QueryModel):

    embeddings = container.model.encode(
        data.query,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT for cosine similarity
    )

    return embeddings.tolist()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_status": container.status,
        "model_name": MODEL_NAME
    }

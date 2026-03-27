from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
from uuid import UUID
import os


app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

class Chunk(BaseModel):
    id: UUID
    file_path: str
    start_line: int
    end_line: int
    content: str

class QueryModel(BaseModel):
    query: str    


@app.post("/encode-chunks")
def encode_chunks(chunks: List[Chunk], batch_size=32):
    
    texts = [chunk.content for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT for cosine similarity
    )

    return embeddings.tolist()


@app.post("/encode-query")
def encode_query(data: QueryModel):

    embeddings = model.encode(
        data.query,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT for cosine similarity
    )

    return embeddings.tolist()
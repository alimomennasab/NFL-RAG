import os
import json
from typing import List, Dict
import faiss
import numpy as np
import dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

INDEX_PATH = "store/index.faiss"
CHUNKS_PATH = "store/chunks.jsonl"
dotenv.load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
TOP_K = 3
llm = None
index = None
chunks = None
pc = None

"""
Function for loading the LLM from sentence transformers
"""
def load_llm():
    global llm
    if llm is None:
        llm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return llm

"""
Function for loading Pinecone index
"""
def load_index():
    global pc, index
    if index is not None:
        return index
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    index = pc.Index(INDEX_NAME)
    return index

"""
Function for embedding a list of text strings into vectors
"""
def embed_texts(texts: List[str]):
    X = load_llm().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return X.astype("float32") 

"""
Function for retrieving the top k most relevant chunks for a query
"""
def retrieve(query: str, k: int = TOP_K):
    index = load_index()
    embedded_vector = embed_texts([query])[0].tolist()
    res = index.query(vector=embedded_vector, top_k=k, include_metadata=True)

    out = []
    for rank, m in enumerate(res.matches):
        out.append({
            "chunk_id": m.id,
            "text": m.metadata["text"],
            "rank": rank,
            "score": m.score,
        })
    return out

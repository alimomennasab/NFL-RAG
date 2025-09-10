import os, json
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "store/index.faiss"
CHUNKS_PATH = "store/chunks.jsonl"
TOP_K = 3
llm = None
index = None
chunks = None

"""
Function for loading the LLM from sentence transformers
"""
def load_llm():
    global llm
    if llm is None:
        llm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return llm

"""
Function for loading FAISS index
"""
def load_index():
    global index
    if index is None:
        index = faiss.read_index(INDEX_PATH)
    return index

"""
Function for loading chunks from a JSONL file
"""
def load_chunks():
    global chunks
    if chunks is None:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
    return chunks

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
    qv = embed_texts([query])
    D, I = load_index().search(qv, k)
    out: List[Dict] = []
    chs = load_chunks()
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < 0:
            continue
        item = dict(chs[idx])
        item["rank"] = rank
        item["score"] = float(score)
        out.append(item)
    return out

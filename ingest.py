import os
import json
import re
import faiss
import pandas as pd
import dotenv
from embed import embed_texts
from pathlib import Path
from typing import List
from pinecone import Pinecone, ServerlessSpec

DATA_DIR = Path("data")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
INDEX_PATH = "store/index.faiss"
CHUNKS_PATH = "store/chunks.jsonl"

dotenv.load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")

"""
Return entire text of a .txt file as a string
"""
def read_txt(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore")

"""
Return entire text of a .csv file as a string
"""
def read_csv(path: Path):
    df = pd.read_csv(path)
    return "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())

"""
Load all documents from the data directory
"""
def load_docs(data_dir: Path):
    docs = []
    for p in data_dir.rglob("*"):
        if p.is_dir():
            continue
        elif p.suffix.lower() == ".txt":
            text = read_txt(p)
        elif p.suffix.lower() == ".csv":
            text = read_csv(p)
        else:
            continue
        if text.strip():
            docs.append({"path": str(p), "text": text})

    return docs

"""
Chunk a text string into smaller pieces with overlap
"""
def chunk(text: str, size: int, overlap: int):
    words = re.split(r"\s+", text)
    chunks = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i+size]).strip()
        if piece:
            chunks.append(piece)
        i += max(1, size - overlap)
    return chunks

if __name__ == "__main__":
    docs = load_docs(DATA_DIR)
    print(f"Loaded {len(docs)} docs")

    all_chunks = []
    # create chunks from all docs
    for d in docs:
        for ch in chunk(d["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_id = f"c{len(all_chunks)}"
            all_chunks.append({"chunk_id": chunk_id, "text": ch, "source": d["path"]})

    print(f"Created {len(all_chunks)} chunks")

    # embed every chunk's text value 
    X = embed_texts([c["text"] for c in all_chunks])
    print("shape of embeddings: ", X.shape)
    dim = X.shape[1] # length of an embedding vector

    # debug: store embedded chunks to the JSONL file
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # upsert vectors to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=X.shape[1], # MiniLM-L6-v2 dims are 384
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    pindex = pc.Index(INDEX_NAME)

    vectors = [
        {
            "id": c["chunk_id"],
            "values": X[i].tolist(),
            "metadata": {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c.get("source", "unknown"),
            },
        }
        for i, c in enumerate(all_chunks)
    ]
    pindex.upsert(vectors=vectors)

    print(f"Upserted {len(all_chunks)} vectors to Pinecone index '{INDEX_NAME}'.")

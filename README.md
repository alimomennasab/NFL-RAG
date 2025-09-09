data preprocessing:
1. load data docs
2. chunk docs
3. embed chunks (transformer/llama?) 
4. place chunks in vector db, indexed with faiss

data querying:
1. encode input question 
2. retrieve nearest neighbor search the encoded question in vector db with faiss
3. use gemini api with input question and retrieved context. 
    - specify to only use the retrieved context to answer the query
4. 
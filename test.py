import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index_path = "faiss_index.bin"
df_path = "processed_products.json"

def load_index(index_path):
    index = faiss.read_index(index_path)
    return index

def load_data(df_path):
    with open(df_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def find_similar_products(query, df, index, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [df[i] for i in indices[0]]

if __name__ == "__main__":
    index = load_index(index_path)
    df = load_data(df_path)
    
    test_query = "Cotton baby bib with cute designs"
    results = find_similar_products(test_query, df, index)
    
    print("Top similar products:")
    for r in results:
        print(json.dumps(r, indent=2, ensure_ascii=False))
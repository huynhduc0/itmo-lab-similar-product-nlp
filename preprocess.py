import json
import os
import gzip
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
PROCESSED_FILE = "processed_products.json"
FAISS_INDEX_FILE = "faiss_index.bin"

def load_data_from_gz(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".gz"):
            file_path = os.path.join(folder, filename)
            print(f"Processing: {filename}")
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        product = json.loads(line)
                        item = {
                            "item_id": product.get("item_id", ""),
                            "item_name": product.get("item_name", [{}])[0].get("value", ""),
                            "brand": product.get("brand", [{}])[0].get("value", ""),
                            "color": product.get("color", [{}])[0].get("value", ""),
                            "product_type": product.get("product_type", [{}])[0].get("value", ""),
                            "description": product.get("bullet_point", [{}])[0].get("value", ""),
                        }
                        item["full_text"] = f"{item['item_name']} {item['brand']} {item['color']} {item['product_type']} {item['description']}"
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
    return data

data = load_data_from_gz(DATA_FOLDER)
descriptions = [item["full_text"] for item in data]
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
embeddings = model.encode(descriptions, convert_to_tensor=False)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_FILE)

with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Training complete! Saved {len(data)} products to {PROCESSED_FILE} and FAISS index to {FAISS_INDEX_FILE}.")
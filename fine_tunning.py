import os
import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ==== CONFIG ====
MODEL_PATH = "fine_tuned_model"
FAISS_INDEX_FILE = "faiss_index.bin"
PRODUCTS_FILE = "processed_products.json"

# ==== LOAD DATA ====
with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📦 Loaded {len(data)} products.")

# ==== CREATE TRAINING EXAMPLES ====
train_examples = [InputExample(texts=[item["full_text"], item["full_text"]]) for item in data]
print(f"✅ Training examples created: {len(train_examples)}")

# ==== LOAD OR TRAIN MODEL ====
if os.path.exists(MODEL_PATH):
    print(f"🚀 Loading fine-tuned model from {MODEL_PATH}...")
    model = SentenceTransformer(MODEL_PATH)
else:
    print("🚀 Fine-tuning model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Tạo dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Train model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

    # Lưu model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

# ==== BUILD FAISS INDEX ====
print("🔄 Encoding product descriptions...")
embeddings = model.encode([item["full_text"] for item in data], convert_to_numpy=True)
dimension = embeddings.shape[1]

print("🛠️ Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Lưu FAISS index
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"✅ FAISS index saved at {FAISS_INDEX_FILE}")

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
import faiss
import json
from fastapi.templating import Jinja2Templates
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
from fastapi.staticfiles import StaticFiles
from typing import Optional
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Image base URL (thay bằng domain của bạn nếu cần)
BASE_URL = "https://m.media-amazon.com/images/I/"

# Load product data
with open("processed_products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract fields
product_descriptions = [item.get("full_text", item["description"]) for item in data]
product_name = [item["item_name"] for item in data]
product_ids = [item["item_id"] for item in data]
product_images = [item["main_image_id"] for item in data]
product_colors = [item.get("color", "Unknown") for item in data]

# Load Sentence Transformer model
tokenized_corpus = [nltk.word_tokenize(desc.lower()) for desc in product_descriptions]
bm25 = BM25Okapi(tokenized_corpus)
MODEL_PATH = "fine_tuned_model"
FAISS_INDEX_FILE = "faiss_index_finetuned.bin"
templates = Jinja2Templates(directory="templates")
if os.path.exists(MODEL_PATH):
    print("🔹 Loading fine-tuned model...")
    model = SentenceTransformer(MODEL_PATH)
else:
    print("⚠️ Fine-tuned model not found! Using default model.")
    model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
if os.path.exists(FAISS_INDEX_FILE):
    print("🚀 Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("❌ FAISS index not found!")
    index = None  # Set index to None if not found


# Placeholder cho BM25 Search (Bạn cần thay bằng implement thực tế)
def bm25_search(query, top_k=5):
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "item_id": product_ids[idx],
            "item_name": product_name[idx],
            "image_url": f"{BASE_URL}{product_images[idx]}.jpg",
            "description": product_descriptions[idx],
            "color": product_colors[idx],
            "bm25_score": float(scores[idx])
        })
    return results


class ResultItem(BaseModel):
    item_name: str
    item_id: str
    description: str
    image_url: str
    color: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[ResultItem]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search/", response_model=SearchResponse)
def search_similar_products(
    query: str = Query(..., title="Product Description"),
    top_k: int = Query(5, title="Number of results"),
    min_score: float = Query(0.0, title="Minimum score"),
    color: Optional[str] = Query(None, title="Filter by color"),
    use_hybrid: bool = Query(False, title="Use Hybrid Search")
):
    """Search for similar products based on query"""
    results = []

    if use_hybrid and index is not None:  # Check index tồn tại
        # Chạy BM25 trước
        bm25_results = bm25_search(query, top_k * 2)
        # Chạy FAISS
        query_embedding = model.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(
            query_embedding.reshape(1, -1), top_k * 2
        )  # Reshape query_embedding

        # Normalize BM25 scores and Faiss distances
        max_bm25_score = (
            max([r["bm25_score"] for r in bm25_results]) if bm25_results else 1.0
        )
        max_faiss_distance = (
            np.max(distances) if distances.size > 0 else 1.0
        )  # Handle empty distances

        # Kết hợp kết quả BM25 và FAISS
        combined_results = []
        for i, idx in enumerate(indices[0]):
            # Find the bm25 result corresponding to faiss index
            bm25_result = next(
                (r for r in bm25_results if r["item_id"] == product_ids[idx]), None
            )
            bm25_score = bm25_result["bm25_score"] if bm25_result else 0.0

            faiss_distance = distances[0][i]

            # Normalize scores
            normalized_bm25_score = (
                bm25_score / max_bm25_score if max_bm25_score > 0 else 0.0
            )
            normalized_faiss_distance = (
                faiss_distance / max_faiss_distance if max_faiss_distance > 0 else 0.0
            )

            # Hybrid Score Calculation:  Adjust weights to prioritize BM25 or FAISS
            bm25_weight = 0.6
            faiss_weight = 0.4
            hybrid_score = (
                bm25_weight * normalized_bm25_score
                + faiss_weight * (1 - normalized_faiss_distance)
            )  # Higher distance means less relevant, so subtract from 1

            combined_results.append(
                {
                    "item_name": product_name[idx],
                    "item_id": product_ids[idx],
                    "description": product_descriptions[idx],
                    "image_url": f"{BASE_URL}{product_images[idx]}.jpg",
                    "color": product_colors[idx],
                    "score": hybrid_score,
                }
            )

        # Sắp xếp theo hybrid_score giảm dần
        results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    else:
        # Chỉ dùng BM25 nếu không bật Hybrid Search hoặc không có FAISS index
        bm25_results = bm25_search(query, top_k)
        results = [
            {
                "item_name": r["item_name"],
                "item_id": r["item_id"],
                "description": r["description"],
                "image_url": r["image_url"],
                "color": r["color"],
                "score": r["bm25_score"],
            }
            for r in bm25_results
        ]

    # Filter by color
    if color:
        results = [r for r in results if r["color"].lower() == color.lower()]

    # Filter by minimum score
    results = [r for r in results if r["score"] >= min_score]

    # Take top_k after filtering
    results = results[:top_k]

    return {"query": query, "results": [ResultItem(**r) for r in results]}
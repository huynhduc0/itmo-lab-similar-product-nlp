from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from jinja2 import Template

app = FastAPI()

# Image base URL (replace with your actual domain)
BASE_URL = "https://example.com/images/"

# Load product data
with open("processed_products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract fields
product_descriptions = [item["description"] for item in data]
product_ids = [item["item_id"] for item in data]
product_images = [item["main_image_id"] for item in data]
# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(product_descriptions, convert_to_tensor=False)
dimension = embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print("heheheheheeheheh")
class SearchResponse(BaseModel):
    query: str
    results: list

@app.get("/", response_class=HTMLResponse)
async def home():
    """Landing page with search form"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Product Search</title>
        <link rel="stylesheet" 
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            .loading-spinner {
                display: none;
                width: 3rem;
                height: 3rem;
                margin: 10px auto;
            }
            .product-card img {
                max-height: 200px;
                object-fit: cover;
                border-radius: 10px;
            }
        </style>
    </head>
    <body class="bg-light">
        <div class="container text-center mt-5">
            <h2 class="mb-4">🔍 Find Similar Products</h2>
            <div class="input-group mb-3">
                <input type="text" id="query" class="form-control" placeholder="Enter product description...">
                <button class="btn btn-primary" onclick="search()">Search</button>
            </div>
            <div class="text-center">
                <div class="spinner-border text-primary loading-spinner" role="status"></div>
            </div>
            <div id="results" class="row mt-4"></div>
        </div>

        <script>
            function search() {
                let query = document.getElementById('query').value;
                if (!query) return;

                $(".loading-spinner").show();
                $("#results").html("");

                fetch(`/search/?query=${encodeURIComponent(query)}`)
                    .then(response => response.json())
                    .then(data => {
                        $(".loading-spinner").hide();
                        let html = "";
                        data.results.forEach(product => {
                            html += `
                                <div class="col-md-4">
                                    <div class="card product-card mb-3">
                                        <img src="${product.image_url}" class="card-img-top" alt="Product Image">
                                        <div class="card-body">
                                            <p class="card-text">${product.description}</p>
                                        </div>
                                    </div>
                                </div>`;
                        });
                        $("#results").html(html);
                    });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/search/", response_model=SearchResponse)
def search_similar_products(query: str = Query(..., title="Product Description"), top_k: int = 5):
    """Search for similar products based on description"""
    query_embedding = model.encode([query], convert_to_tensor=False)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "item_id": product_ids[idx],
            "description": product_descriptions[idx],
            "image_url": f"{BASE_URL}{product_images[idx]}.jpg"
        })

    return {"query": query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

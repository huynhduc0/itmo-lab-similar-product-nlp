function toggleHybridSearch() {
    let isHybrid = document.getElementById("use_hybrid").checked;
    document.getElementById("max_distance").disabled = isHybrid;
}

function search() {
    let query = document.getElementById('query').value;
    let top_k = document.getElementById('top_k').value;
    let max_distance = document.getElementById('max_distance').value;
    let color = document.getElementById('color').value;
    let use_hybrid = document.getElementById('use_hybrid').checked;

    if (!query) return;

    $(".loading-spinner").show();
    $("#results").html("");

    let url = `/search/?query=${encodeURIComponent(query)}&top_k=${top_k}&max_distance=${max_distance}&color=${encodeURIComponent(color)}&use_hybrid=${use_hybrid}`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            $(".loading-spinner").hide();
            let html = "";
            data.results.forEach(product => {
                let scoreText = use_hybrid
                    ? `Hybrid Score: ${product.hybrid_score.toFixed(4)}`
                    : `BM25 Score: ${product.bm25_score.toFixed(2)}`;

                html += `
                    <div class="col-md-4">
                        <div class="card product-card mb-3">
                            <img src="${product.image_url}" class="card-img-top" alt="Product Image">
                            <div class="card-body">
                                <p class="card-text">${product.item_name}</p>
                                <p class="text-muted">Color: ${product.color} | ${scoreText}</p>
                                <button class="btn btn-info" onclick='viewProduct(${JSON.stringify(product)})'>View Details</button>
                            </div>
                        </div>
                    </div>`;
            });
            $("#results").html(html);
        });
}

function viewProduct(product) {
    $("#modalTitle").text(product.item_name);
    $("#modalBody").html(`
        <img src="${product.image_url}" alt="Product Image">
        <p>Color: ${product.color}</p>
        <p>Item ID: ${product.item_id}</p>
        <p>Description: ${product.description}</p>
    `);
    $("#productModal").modal('show');
}

from client import TopKHeapSelector

vectors = {
    "doc1": [1.0, 0.0, 0.0],
    "doc2": [0.8, 0.2, 0.0],
    "doc3": [0.0, 1.0, 0.0]
}
top = TopKHeapSelector.select_topk([1.0, 0.0, 0.0], vectors, k=2)
print("Top-2 Exact Cosine Matches:", top)

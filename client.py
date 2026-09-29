"""Vector Similarity Exact Top-K Heap Selector.
100% Python Standard Library.
"""

import math
import heapq

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def norm(v):
    return math.sqrt(sum(x * x for x in v)) or 1e-9

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (norm(v1) * norm(v2))

def euclidean_dist(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class TopKHeapSelector:
    """Fast bounded Min-Heap Top-K cosine similarity selector."""
    @staticmethod
    def select_topk(query_vec, candidate_vectors, k=3, metric="cosine"):
        heap = []
        for nid, v in candidate_vectors.items():
            if metric == "cosine":
                score = cosine_similarity(query_vec, v)
            else:
                score = -euclidean_dist(query_vec, v)

            if len(heap) < k:
                heapq.heappush(heap, (score, nid))
            else:
                if score > heap[0][0]:
                    heapq.heappushpop(heap, (score, nid))

        results = []
        while heap:
            score, nid = heapq.heappop(heap)
            results.append({"id": nid, "score": round(score if metric == "cosine" else -score, 4)})
        results.reverse()
        return results

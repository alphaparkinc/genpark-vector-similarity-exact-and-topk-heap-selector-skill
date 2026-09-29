# genpark-vector-similarity-exact-and-topk-heap-selector-skill

Bounded Min-Heap Top-K vector selector for exact cosine similarity and Euclidean distance ranking.

## Architecture

```mermaid
flowchart TD
    Candidates[N Candidate Vectors] --> Scoring[Similarity Scoring Engine]
    Scoring --> Heap["Bounded Min-Heap (Size K)"]
    Heap --> Prune[Prune Lowest Score When Full]
    Heap --> Sorted[Top-K Descending Output]
```

## Features
- **O(N log K) Complexity**: Fixed memory footprint without sorting full array.
- **Metric Options**: Supports both cosine similarity and Euclidean distance.

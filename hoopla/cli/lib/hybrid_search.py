import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # Get results from both searches (500x limit for better coverage)
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Extract scores for normalization
        bm25_scores = [r["score"] for r in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]

        # Normalize scores
        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        # Create document score dictionary
        doc_scores = {}

        # Add BM25 scores
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            doc_scores[doc_id] = {
                "document": result,
                "bm25_score": normalized_bm25[i],
                "semantic_score": 0.0,
            }

        # Add semantic scores
        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["semantic_score"] = normalized_semantic[i]
            else:
                doc_scores[doc_id] = {
                    "document": result,
                    "bm25_score": 0.0,
                    "semantic_score": normalized_semantic[i],
                }

        # Calculate hybrid scores
        for doc_id in doc_scores:
            bm25 = doc_scores[doc_id]["bm25_score"]
            semantic = doc_scores[doc_id]["semantic_score"]
            hybrid = alpha * bm25 + (1 - alpha) * semantic
            doc_scores[doc_id]["hybrid_score"] = hybrid

        # Sort by hybrid score
        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
        )

        # Format results
        results = []
        for doc_id, scores in sorted_results[:limit]:
            doc = scores["document"]
            description = doc.get("document") or doc.get("description", "")
            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "description": description,
                    "hybrid_score": scores["hybrid_score"],
                    "bm25_score": scores["bm25_score"],
                    "semantic_score": scores["semantic_score"],
                }
            )

        return results

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        doc_scores = {}

        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["id"]
            rrf_score = 1 / (k + rank)
            doc_scores[doc_id] = {
                "document": result,
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score,
            }
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result["id"]
            rrf_score = 1 / (k + rank)
            if doc_id in doc_scores:
                doc_scores[doc_id]["semantic_scores"] = rank

                doc_scores[doc_id]["rrf_score"] += rrf_score
            else:
                doc_scores[doc_id] = {
                    "document": result,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score,
                }
        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        results = []
        for doc_id, scores in sorted_results[:limit]:
            doc = scores["document"]
            description = doc.get("document") or doc.get("description", "")
            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "description": description,
                    "rrf_score": scores["rrf_score"],
                    "bm25_rank": scores["bm25_rank"],
                    "semantic_rank": scores["semantic_rank"],
                }
            )
        return results


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)
    normalized = []
    for score in scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append(norm_score)
    return normalized


def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5) -> None:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    results = hybrid.weighted_search(query, alpha, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
        print(
            f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}"
        )
        print(f"   {result['description'][:100]}...")
        print()


def rrf_search_command(query: str, k: int = 60, limit: int = 5) -> None:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    results = hybrid.rrf_search(query, k, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        bm25_rank = result["bm25_rank"] if result["bm25_rank"] else "N/A"
        semantic_rank = result["semantic_rank"] if result["semantic_rank"] else "N/A"
        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
        print(f"   {result['description'][:100]}...")
        print()

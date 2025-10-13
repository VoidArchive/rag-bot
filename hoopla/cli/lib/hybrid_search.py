import os
import time

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


def rrf_search_command(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str = None,
    rerank_method: str = None,
) -> None:
    # Enhance query if requested
    original_query = query
    if enhance == "spell":
        query = enhance_query_spell(query)
        if query != original_query:
            print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")
    elif enhance == "rewrite":
        query = enhance_query_rewrite(query)
        if query != original_query:
            print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")
    elif enhance == "expand":
        query = enhance_query_expand(query)
        if query != original_query:
            print(f"Enhanced query ({enhance}): '{original_query}' -> '{query}'\n")

    documents = load_movies()
    hybrid = HybridSearch(documents)

    # Get more results if reranking
    search_limit = limit * 5 if rerank_method == "individual" else limit
    results = hybrid.rrf_search(query, k, search_limit)

    # Rerank if requested
    if rerank_method == "individual":
        print(f"Reranking top {len(results)} results using individual method...")
        results = rerank_individual(query, results)
        results = results[:limit]  # Truncate to original limit
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
            print(f"   RRF Score: {result['rrf_score']:.3f}")
            bm25_rank = result["bm25_rank"] if result["bm25_rank"] else "N/A"
            semantic_rank = (
                result["semantic_rank"] if result["semantic_rank"] else "N/A"
            )
            print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
            print(f"   {result['description'][:100]}...")
            print()
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   RRF Score: {result['rrf_score']:.3f}")
            bm25_rank = result["bm25_rank"] if result["bm25_rank"] else "N/A"
            semantic_rank = (
                result["semantic_rank"] if result["semantic_rank"] else "N/A"
            )
            print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
            print(f"   {result['description'][:100]}...")
            print()


def enhance_query_spell(query: str) -> str:
    from google import genai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )
    if response.text is not None:
        return response.text.strip()
    else:
        return "Gemini is not working"


def enhance_query_rewrite(query: str) -> str:
    from google import genai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )
    if response.text is not None:
        return response.text.strip()
    else:
        return "Gemini is not working"


def enhance_query_expand(query: str) -> str:
    from google import genai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )
    if response.text is not None:
        return response.text.strip()
    else:
        return "Gemini is not working"


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    from google import genai
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    reranked_results = []
    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
        )
        score = 0

        if response.text is not None:
            score = float(response.text.strip())
        else:
            score = 5.0
        doc["reranked_score"] = score
        reranked_results.append(doc)
        time.sleep(3)
    reranked_results.sort(key=lambda x: x["reranked_score"], reverse=True)
    return reranked_results

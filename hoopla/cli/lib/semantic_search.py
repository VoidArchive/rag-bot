import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = id
        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((score, self.documents[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return results


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(
        f"embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_movies(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)

    for i, result in enumerate(results, 1):
        desc = (
            result["description"][:100] + "..."
            if len(result["description"]) > 100
            else result["description"]
        )
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"    {desc}\n")


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 1) -> None:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + max_chunk_size])
        chunks.append(chunk)
        i += max_chunk_size - overlap

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

import os

from huggingface_hub import duplicate_space
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies


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

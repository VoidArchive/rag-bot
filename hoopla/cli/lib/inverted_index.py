import os
import pickle
from collections import defaultdict

from .keyword_search import tokenize_text
from .search_utils import load_movies


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self) -> None:
        movies = load_movies()
        for idx, movie in enumerate(movies, start=1):
            doc_id = idx
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)

        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        index_path = os.path.join("cache", "index.pkl")
        docmap_path = os.path.join("cache", "docmap.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError(
                "Index files not found. Run 'Run 'build' command first."
            )
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

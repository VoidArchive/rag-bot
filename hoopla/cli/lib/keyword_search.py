import string

from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_stopwords,
)


def search_command(index, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    query_tokens = tokenize_text(query)
    results = []
    seen_ids = set()
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            results.append(index.docmap[doc_id])
            seen_ids.add(doc_id)
            if len(results) >= limit:
                return results
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    stop_words = load_stopwords()
    stemmer = PorterStemmer()

    return [
        stemmer.stem(token)
        for token in text.split()
        if token and token not in stop_words
    ]

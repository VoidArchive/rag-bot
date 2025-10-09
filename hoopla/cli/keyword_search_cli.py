#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_movies(args.query)

        case _:
            parser.print_help()


def search_movies(query: str) -> None:
    with open("data/movies.json", "r") as f:
        data = json.load(f)

    with open("data/stopwords.txt", "r") as f:
        stop_words = f.read().splitlines()

    translator = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()
    query_normalized = query.lower().translate(translator)
    query_tokens = [
        stemmer.stem(token)
        for token in query_normalized.split()
        if token and token not in stop_words
    ]

    results = []
    for movie in data["movies"]:
        title_normalized = movie["title"].lower().translate(translator)
        title_tokens = [
            stemmer.stem(token)
            for token in title_normalized.split()
            if token and token not in stop_words
        ]
        if any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
        ):
            results.append(movie)

    results.sort(key=lambda m: m["id"])
    results = results[:5]

    print(f"Searching for: {query}")
    for i, movie in enumerate(results, start=1):
        print(f"{i}, {movie['title']}")


if __name__ == "__main__":
    main()

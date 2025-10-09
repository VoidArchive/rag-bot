#!/usr/bin/env python3

import argparse
import json
import string


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

    translator = str.maketrans("", "", string.punctuation)

    results = []
    query_normalized = query.lower().translate(translator)
    for movie in data["movies"]:
        title_normalized = movie["title"].lower().translate(translator)
        if query_normalized in title_normalized:
            results.append(movie)

    results.sort(key=lambda m: m["id"])
    results = results[:5]

    print(f"Searching for: {query}")
    for i, movie in enumerate(results, start=1):
        print(f"{i}, {movie['title']}")


if __name__ == "__main__":
    main()

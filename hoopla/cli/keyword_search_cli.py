#!/usr/bin/env python3

import argparse
import sys

from lib.inverted_index import InvertedIndex
from lib.keyword_search import (
    search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search")

    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(f"Searching for: {args.query}")
            results = search_command(index, args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}, {res['title']}")

        case "build":
            index = InvertedIndex()
            index.build()
            index.save()

        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

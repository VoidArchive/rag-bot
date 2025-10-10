#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_text,
    search_movies,
    semantic_chunk_text,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    embed_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the embedding model")
    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for query"
    )
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk text into fixed-size pieces"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text on sentence boundaries"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Number of sentences to overlap between chunks",
    )
    subparsers.add_parser(
        "embed_chunks", help="Generate embeddings for document chunks"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_movies(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.exit(2, parser.print_help())


if __name__ == "__main__":
    main()

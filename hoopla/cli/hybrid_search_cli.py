import argparse

from lib.hybrid_search import (
    normalize_scores,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores to normalize"
    )
    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Search using weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for BM25 score (default: 0.5)"
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )
    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Search using Reciprocal Rank Fusion"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "--k", type=int, default=60, help="RRF k parameter (default: 60)"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Reranking method to use",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score: .4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance)
        case "rrf-search":
            rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the embedding model")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.exit(2, parser.print_help())


if __name__ == "__main__":
    main()

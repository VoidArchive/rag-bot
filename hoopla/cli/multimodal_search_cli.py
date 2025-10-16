import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify CLIP model and image embedding generation"
    )
    verify_parser.add_argument("--image", required=True, type=str, help="Path to image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image"
    )
    image_search_parser.add_argument("--image", required=True, type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            result = verify_image_embedding_command(args.image)
            print(f"Embedding shape: {result['embedding_shape']} dimensions")

        case "image_search":
            results = image_search_command(args.image)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
                # Print first 100 characters of description
                desc = result['description'][:100]
                if len(result['description']) > 100:
                    desc += "..."
                print(f"   {desc}")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

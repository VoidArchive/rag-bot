import argparse

from lib.describe_image import describe_image_command


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal query rewriting using Gemini"
    )
    parser.add_argument("--image", required=True, type=str, help="Path to image file")
    parser.add_argument(
        "--query", required=True, type=str, help="Text query to rewrite"
    )

    args = parser.parse_args()

    result = describe_image_command(args.image, args.query)

    print(f"Rewritten query: {result['rewritten_query']}")
    if result["total_tokens"] is not None:
        print(f"Total tokens:    {result['total_tokens']}")


if __name__ == "__main__":
    main()

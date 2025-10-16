import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def describe_image_command(image_path: str, query: str) -> dict:
    """
    Rewrite a text query based on an image using Gemini's multimodal capabilities.

    Args:
        image_path: Path to the image file
        query: Text query to rewrite

    Returns:
        Dictionary containing the rewritten query and token usage
    """
    # Determine MIME type
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    # Read image file
    with open(image_path, "rb") as f:
        img = f.read()

    # System prompt
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    # Build request parts
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    # Send query to Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp", contents=parts
    )

    return {
        "rewritten_query": response.text.strip(),
        "total_tokens": (
            response.usage_metadata.total_token_count
            if response.usage_metadata
            else None
        ),
    }

from sentence_transformers import SentenceTransformer

from .search_utils import load_movies
from .semantic_search import cosine_similarity


def verify_image_embedding_command(image_path: str) -> dict:
    """
    Load a CLIP model and generate an image embedding to verify functionality.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing embedding shape information
    """
    # Load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    # Generate embedding for the image
    # encode() expects a list and returns a list of embeddings
    embeddings = model.encode([image_path])

    # Get the first (and only) embedding
    embedding = embeddings[0]

    return {"embedding_shape": embedding.shape[0]}


def image_search_command(image_path: str, limit: int = 5) -> list[dict]:
    """
    Search for movies using an image query with CLIP embeddings.

    Args:
        image_path: Path to the image file
        limit: Number of results to return (default=5)

    Returns:
        List of search results with similarity scores
    """
    # Load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    # Generate embedding for the image
    image_embedding = model.encode([image_path])[0]

    # Load movie database
    documents = load_movies()

    # Create text representations combining title and description
    texts = []
    for doc in documents:
        texts.append(f"{doc['title']}: {doc['description']}")

    # Generate text embeddings for all movies
    text_embeddings = model.encode(texts, show_progress_bar=True)

    # Calculate cosine similarity between image and each text embedding
    similarities = []
    for i, text_embedding in enumerate(text_embeddings):
        similarity = cosine_similarity(image_embedding, text_embedding)
        similarities.append(
            {
                "title": documents[i]["title"],
                "description": documents[i]["description"],
                "similarity": float(similarity),
            }
        )

    # Sort by similarity (highest first) and return top results
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:limit]

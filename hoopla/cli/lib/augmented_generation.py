from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import load_movies, RRF_K

load_dotenv()


def rag_command(query: str) -> dict:
    """Perform RAG: search for relevant documents and generate an answer using Gemini.

    Args:
        query: The search query

    Returns:
        Dictionary containing query, results, and generated response
    """
    # Load movies and perform RRF search
    movies = load_movies()
    searcher = HybridSearch(movies)

    # Search for top 5 results
    results = searcher.rrf_search(query, k=RRF_K, limit=5)

    # Format search results for the prompt
    docs_text = ""
    for result in results:
        docs_text += f"Title: {result['title']}\nDescription: {result['document']}\n\n"

    # Create prompt for Gemini
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs_text}

Provide a comprehensive answer that addresses the query:"""

    # Call Gemini API
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    return {
        "query": query,
        "results": results,
        "response": response.text,
    }


def summarize_command(query: str, limit: int = 5) -> dict:
    """Perform multi-document summarization of search results.

    Args:
        query: The search query
        limit: Number of search results to retrieve (default: 5)

    Returns:
        Dictionary containing query, results, and generated summary
    """
    # Load movies and perform RRF search
    movies = load_movies()
    searcher = HybridSearch(movies)

    # Search for results
    results = searcher.rrf_search(query, k=RRF_K, limit=limit)

    # Format search results for the prompt
    results_text = ""
    for result in results:
        results_text += f"Title: {result['title']}\nDescription: {result['document']}\n\n"

    # Create prompt for Gemini
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{results_text}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    # Call Gemini API
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    return {
        "query": query,
        "limit": limit,
        "results": results,
        "summary": response.text,
    }


def citations_command(query: str, limit: int = 5) -> dict:
    """Generate an answer with citations referencing the source documents.

    Args:
        query: The search query
        limit: Number of search results to retrieve (default: 5)

    Returns:
        Dictionary containing query, results, and generated answer with citations
    """
    # Load movies and perform RRF search
    movies = load_movies()
    searcher = HybridSearch(movies)

    # Search for results
    results = searcher.rrf_search(query, k=RRF_K, limit=limit)

    # Format search results for the prompt with numbered citations
    documents = ""
    for i, result in enumerate(results, 1):
        documents += f"[{i}] {result['title']}\n{result['document']}\n\n"

    # Create prompt for Gemini
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    # Call Gemini API
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    return {
        "query": query,
        "limit": limit,
        "results": results,
        "answer": response.text,
    }


def question_command(question: str, limit: int = 5) -> dict:
    """Answer a user's question in a conversational, casual manner.

    Args:
        question: The user's question
        limit: Number of search results to retrieve (default: 5)

    Returns:
        Dictionary containing question, results, and generated answer
    """
    # Load movies and perform RRF search
    movies = load_movies()
    searcher = HybridSearch(movies)

    # Search for results
    results = searcher.rrf_search(question, k=RRF_K, limit=limit)

    # Format search results for the prompt
    context = ""
    for result in results:
        context += f"Title: {result['title']}\nDescription: {result['document']}\n\n"

    # Create prompt for Gemini
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    # Call Gemini API
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    return {
        "question": question,
        "limit": limit,
        "results": results,
        "answer": response.text,
    }

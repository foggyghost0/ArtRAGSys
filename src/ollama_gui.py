"""
Simplified Ollama integration module for GUI usage.
Provides essential RAG capabilities for the ArtRAG System GUI.
"""

import requests
from typing import List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from retrieval_gui import ThreadSafeArtSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result for easier handling."""

    artwork_id: str
    title: str
    author: str
    content: str
    search_type: str
    relevance_score: float
    image_path: Optional[str] = None


class OllamaArtRAGGUI:
    """
    Simplified Ollama Art RAG system for GUI usage.
    Focuses on essential functionality needed for the interface.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "gemma3:4b-it-qat",
        db_path: str = "art_database.db",
        chroma_path: str = "./chroma_db",
        image_base_path: str = "./data/img",
    ):
        """Initialize the simplified Ollama Art RAG system."""
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.image_base_path = Path(image_base_path)

        # Initialize the retrieval system
        self.art_search = ThreadSafeArtSearch(db_path=db_path, chroma_path=chroma_path)

        # Verify Ollama connection
        self._verify_ollama_connection()

        logger.info(f"OllamaArtRAGGUI initialized with model: {model_name}")

    def _verify_ollama_connection(self) -> bool:
        """Verify connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
            return True
        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
            logger.warning("Chat functionality will be limited without Ollama")
            return False

    def search_artworks(
        self,
        query: str,
        k: int = 12,
        search_type: str = "comprehensive",
        include_images: bool = True,
    ) -> List[SearchResult]:
        """
        Search for artworks using the retrieval system.

        Args:
            query: Search query
            k: Number of results to return
            search_type: Type of search ('comprehensive', 'semantic', 'text', 'metadata')
            include_images: Whether to include image paths

        Returns:
            List of SearchResult objects
        """
        try:
            # Choose search method based on type
            if search_type == "comprehensive":
                results = self.art_search.comprehensive_search(
                    query, search_type="comprehensive", k=k
                )
            elif search_type == "semantic":
                results = self.art_search.comprehensive_search(
                    query, search_type="semantic", k=k
                )
            elif search_type == "text":
                # Map to keyword search in the new system
                results = self.art_search.comprehensive_search(
                    query, search_type="keyword", k=k
                )
            elif search_type == "metadata":
                # Use comprehensive search which includes metadata extraction
                results = self.art_search.comprehensive_search(
                    query, search_type="comprehensive", k=k
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                # Determine image path
                image_path = None
                if include_images:
                    artwork_id = result.get("id", "")
                    if artwork_id:
                        # The artwork_id is already the image filename (e.g., "00003-rudolf2.jpg")
                        potential_path = self.image_base_path / artwork_id
                        if potential_path.exists():
                            image_path = str(potential_path)

                search_result = SearchResult(
                    artwork_id=str(result.get("id", "")),
                    title=result.get("title", ""),
                    author=result.get("author", ""),
                    content=result.get("description", ""),
                    search_type=search_type,
                    relevance_score=result.get("relevance_score", 0.0),
                    image_path=image_path,
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Error searching artworks: {e}")
            return []

    def generate_response(
        self,
        query: str,
        context_results: List[SearchResult],
        max_context_length: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response using Ollama with retrieved context.

        Args:
            query: User query
            context_results: Retrieved artwork information
            max_context_length: Maximum context length in characters
            temperature: Generation temperature

        Returns:
            Generated response
        """
        try:
            # Build context from search results
            context_parts = []
            current_length = 0

            for result in context_results:
                if current_length >= max_context_length:
                    break

                context_part = f"""
Artwork: {result.title}
Artist: {result.author}
Content: {result.content}
Relevance: {result.relevance_score:.3f}
---"""

                if current_length + len(context_part) <= max_context_length:
                    context_parts.append(context_part)
                    current_length += len(context_part)

            context = "\n".join(context_parts)

            # Create prompt
            prompt = f"""You are an expert art historian and curator. Use the following artwork information to answer the user's question comprehensively and accurately.

Context from Art Database:
{context}

User Question: {query}

Please provide a detailed, informative response based on the retrieved artwork information. Include specific details about the artworks, artists, and relevant art historical context when available."""

            # Generate response with Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "top_p": 0.9, "top_k": 40},
                },
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "Error generating response")

        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return f"Sorry, I cannot generate a response right now. Please make sure Ollama is running with the {self.model_name} model."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def close(self):
        """Close database connections."""
        if hasattr(self.art_search, "close"):
            self.art_search.close()

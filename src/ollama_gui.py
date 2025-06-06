"""
Ollama integration module for GUI usage.
Provides RAG capabilities for the ArtRAG System GUI.
"""

import requests
from typing import List, Optional, Generator
from pathlib import Path
import logging
import json
import base64
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
    Ollama Art RAG system for GUI usage.
    Focuses on functionality needed for the interface.
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

            # Try to parse the response to ensure it's valid JSON
            version_info = response.json()
            logger.info(f"Successfully connected to Ollama server: {version_info}")

            # Also check if the model is available
            try:
                model_response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                model_response.raise_for_status()
                models_info = model_response.json()

                available_models = [
                    model["name"] for model in models_info.get("models", [])
                ]
                if self.model_name in available_models:
                    logger.info(f"Model {self.model_name} is available")
                else:
                    logger.warning(
                        f"Model {self.model_name} not found. Available models: {available_models}"
                    )

            except Exception as model_check_error:
                logger.warning(
                    f"Could not verify model availability: {model_check_error}"
                )

            return True

        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
            logger.warning("Chat functionality will be limited without Ollama")
            return False
        except ValueError as json_error:
            logger.error(f"Invalid JSON response from Ollama server: {json_error}")
            logger.error(f"Response content: {response.text[:200]}...")
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
            search_type: Type of search ('comprehensive', 'semantic', 'text', 'metadata', 'fuzzy', 'hybrid_scoring')
            include_images: Whether to include image paths

        Returns:
            List of SearchResult objects
        """
        try:
            # Use enhanced_comprehensive_search for all search types
            results = self.art_search.enhanced_comprehensive_search(
                query, search_type=search_type, k=k
            )

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
                    relevance_score=result.get(
                        "hybrid_score", result.get("relevance_score", 0.0)
                    ),
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

            # Prepare images for multimodal input
            images = []
            image_descriptions = []
            for result in context_results:
                if result.image_path and Path(result.image_path).exists():
                    encoded_image = self.encode_image_to_base64(result.image_path)
                    if encoded_image:
                        images.append(encoded_image)
                        image_descriptions.append(
                            f"Image of '{result.title}' by {result.author}"
                        )

            # Create prompt with image awareness
            image_context = ""
            if images:
                image_context = f"\n\nYou can also see {len(images)} image(s) of the artwork(s): {', '.join(image_descriptions)}. Use visual details from the image(s) to enhance your response."

            prompt = f"""You are an expert art historian and curator. Use the following artwork information and any provided images to answer the user's question comprehensively and accurately.

Context from Art Database:
{context}{image_context}

User Question: {query}

Please provide a detailed, informative response based on the retrieved artwork information and visual analysis if images are provided. Include specific details about the artworks, artists, visual elements, composition, style, and relevant art historical context when available.
DO NOT include any information that is not present in the provided context or visible in the images.
Please ensure your response is clear, concise, and relevant to the user's query.
"""

            # Prepare request payload
            request_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "options": {"temperature": temperature, "top_p": 0.9, "top_k": 30},
            }

            # Add images to payload if available
            if images:
                request_payload["images"] = images

            # Generate response with Ollama using streaming
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_payload,
                timeout=250,
                stream=True,  # Enable streaming in requests
            )
            response.raise_for_status()

            # Parse the streaming JSON response
            full_response = ""
            try:
                for line in response.iter_lines():
                    if line:
                        # Each line is a JSON object
                        chunk_data = json.loads(line.decode("utf-8"))

                        # Extract the response text from this chunk
                        chunk_text = chunk_data.get("response", "")
                        full_response += chunk_text

                        # Check if this is the final chunk
                        if chunk_data.get("done", False):
                            break

                return full_response if full_response else "Error generating response"

            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                logger.error(f"Response content: {response.text[:200]}...")
                return "Error: Invalid response format from Ollama server"

        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return f"Sorry, I cannot generate a response right now. Please make sure Ollama is running with the {self.model_name} model."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def generate_response_stream(
        self,
        query: str,
        context_results: List[SearchResult],
        max_context_length: int = 2000,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response using Ollama with retrieved context.

        Args:
            query: User query
            context_results: Retrieved artwork information
            max_context_length: Maximum context length in characters
            temperature: Generation temperature

        Yields:
            str: Each chunk of the generated response as it's received
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

            # Prepare images for multimodal input
            images = []
            image_descriptions = []
            for result in context_results:
                if result.image_path and Path(result.image_path).exists():
                    encoded_image = self.encode_image_to_base64(result.image_path)
                    if encoded_image:
                        images.append(encoded_image)
                        image_descriptions.append(
                            f"Image of '{result.title}' by {result.author}"
                        )

            # Create prompt with image awareness
            image_context = ""
            if images:
                image_context = f"\n\nYou can also see {len(images)} image(s) of the artwork(s): {', '.join(image_descriptions)}. Use visual details from the image(s) to enhance your response."

            prompt = f"""You are an expert art historian and curator. Use the following artwork information and any provided images to answer the user's question comprehensively and accurately.

Context from Art Database:
{context}{image_context}

User Question: {query}

Please provide a detailed, informative response based on the retrieved artwork information and visual analysis if images are provided. Include specific details about the artworks, artists, visual elements, composition, style, and relevant art historical context when available.
DO NOT include any information that is not present in the provided context or visible in the images.
Please ensure your response is clear, concise, and relevant to the user's query.
"""

            # Prepare request payload
            request_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature, "top_p": 0.9, "top_k": 30},
            }

            # Add images to payload if available
            if images:
                request_payload["images"] = images

            # Generate streaming response with Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_payload,
                timeout=250,
                stream=True,
            )
            response.raise_for_status()

            # Stream the response chunks
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode("utf-8").strip()
                        if not line_text:
                            continue

                        chunk_data = json.loads(line_text)
                        chunk_text = chunk_data.get("response", "")

                        if chunk_text:
                            yield chunk_text

                        # Check if this is the final chunk
                        if chunk_data.get("done", False):
                            break

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON chunk: {line_text[:100]}... - Error: {e}"
                        )
                        continue
                    except UnicodeDecodeError as e:
                        logger.warning(f"Skipping invalid Unicode chunk: {e}")
                        continue

        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            yield f"Sorry, I cannot generate a response right now. Please make sure Ollama is running with the {self.model_name} model."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error generating response: {str(e)}"

    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """
        Encode an image file to base64 string for multimodal LLM input.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def close(self):
        """Close database connections."""
        if hasattr(self.art_search, "close"):
            self.art_search.close()

"""
Thread-safe retrieval module for Art RAG System GUI.
Provides semantic and hybrid search capabilities for GUI applications.
"""

import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob


class ThreadSafeArtSearch:
    """Thread-safe art search system with ChromaDB vector search and SQLite metadata."""

    def __init__(
        self, db_path: str = "art_database.db", chroma_path: str = "./chroma_db"
    ):
        """Initialize the search system."""
        print("Initializing thread-safe art search system...")

        self.db_path = db_path
        self.chroma_path = chroma_path
        self._thread_local = threading.local()

        # Load sentence transformer model (this is thread-safe)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize spaCy model (load once, use in threads)
        try:
            self.nlp = spacy.load("en_core_web_trf")
            print("✓ spaCy transformer model loaded successfully")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy small model loaded (fallback)")
            except OSError:
                print("✗ No spaCy model found, comprehensive search disabled")
                self.nlp = None

        # Initialize ChromaDB (this works across threads)
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.sentence_collection = self.chroma_client.get_collection(
                "artwork_sentences"
            )
            self.description_collection = self.chroma_client.get_collection(
                "artwork_descriptions"
            )

            print("Loaded ChromaDB collections:")
            print(f"- Sentences: {self.sentence_collection.count()} items")
            print(f"- Descriptions: {self.description_collection.count()} items")

        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            raise

    def _get_db_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(self.db_path)
        return self._thread_local.conn

    def semantic_search_sentences(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find semantically similar sentences from artwork descriptions."""
        try:
            # Query ChromaDB for similar sentences
            results = self.sentence_collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted_results.append(
                    {
                        "sentence": doc,
                        "artwork_id": metadata["artwork_id"],
                        "title": metadata["title"],
                        "author": metadata["author"],
                        "type": metadata["type"],
                        "school": metadata["school"],
                        "sentence_order": metadata["sentence_order"],
                        "distance": distance,
                        "similarity": 1 - distance,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in semantic sentence search: {e}")
            return []

    def semantic_search_descriptions(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find semantically similar full artwork descriptions."""
        try:
            # Query ChromaDB for similar descriptions
            results = self.description_collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted_results.append(
                    {
                        "description": doc,
                        "artwork_id": metadata["artwork_id"],
                        "title": metadata["title"],
                        "author": metadata["author"],
                        "type": metadata["type"],
                        "school": metadata["school"],
                        "distance": distance,
                        "similarity": 1 - distance,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in semantic description search: {e}")
            return []

    def get_artwork_by_id(self, artwork_id: str) -> Optional[Dict[str, Any]]:
        """Get complete artwork information by image_file (primary key)."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT image_file, author, title, medium, dimensions, date, type, school, 
                       timeframe_start, timeframe_end, full_description
                FROM artworks 
                WHERE image_file = ?
            """,
                (artwork_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],  # image_file as id
                    "image_file": row[0],
                    "author": row[1],
                    "title": row[2],
                    "medium": row[3],
                    "dimensions": row[4],
                    "date": row[5],
                    "type": row[6],
                    "school": row[7],
                    "timeframe_start": row[8],
                    "timeframe_end": row[9],
                    "description": row[10],  # full_description mapped to description
                    # Add these for compatibility
                    "technique": row[3],  # medium mapped to technique
                    "timeframe": f"{row[8]}-{row[9]}" if row[8] and row[9] else row[5],
                    "location": "",  # Not in database
                    "url": "",  # Not in database
                    "form": "",  # Not in database
                    "width": "",  # Could extract from dimensions
                    "height": "",  # Could extract from dimensions
                }
            return None

        except Exception as e:
            print(f"Error getting artwork by ID: {e}")
            return None

    def comprehensive_search(
        self, query: str, search_type: str = "all", k: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive search using multiple strategies.

        Args:
            query: Search query
            search_type: Type of search ("all", "semantic", "keyword", "comprehensive")
            k: Number of results to return
        """
        try:
            if search_type == "semantic":
                return self._semantic_only_search(query, k)
            elif search_type == "keyword":
                return self._keyword_only_search(query, k)
            elif search_type == "comprehensive":
                return self._comprehensive_nlp_search(query, k)
            else:  # "all"
                return self._hybrid_search(query, k)

        except Exception as e:
            print(f"Error in comprehensive search: {e}")
            return []

    def _semantic_only_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Semantic search only."""
        sentence_results = self.semantic_search_sentences(query, k * 2)
        description_results = self.semantic_search_descriptions(query, k * 2)

        # Combine and deduplicate
        artwork_scores = defaultdict(list)

        for result in sentence_results:
            artwork_scores[result["artwork_id"]].append(result["similarity"])

        for result in description_results:
            artwork_scores[result["artwork_id"]].append(result["similarity"])

        # Calculate average scores
        final_scores = []
        for artwork_id, scores in artwork_scores.items():
            avg_score = sum(scores) / len(scores)
            final_scores.append((artwork_id, avg_score))

        # Sort by score and get top k
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Get full artwork details
        results = []
        for artwork_id, score in final_scores[:k]:
            artwork = self.get_artwork_by_id(artwork_id)
            if artwork:
                artwork["relevance_score"] = score
                results.append(artwork)

        return results

    def _keyword_only_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Keyword-based search only."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Simple keyword search across multiple fields
            search_query = f"%{query}%"
            cursor.execute(
                """
                SELECT image_file, author, title, medium, dimensions, date, type, school, 
                       timeframe_start, timeframe_end, full_description
                FROM artworks 
                WHERE title LIKE ? OR author LIKE ? OR full_description LIKE ? 
                   OR type LIKE ? OR school LIKE ? OR medium LIKE ?
                LIMIT ?
            """,
                (
                    search_query,
                    search_query,
                    search_query,
                    search_query,
                    search_query,
                    search_query,
                    k,
                ),
            )

            results = []
            for row in cursor.fetchall():
                artwork = {
                    "id": row[0],  # image_file as id
                    "image_file": row[0],
                    "author": row[1],
                    "title": row[2],
                    "medium": row[3],
                    "dimensions": row[4],
                    "date": row[5],
                    "type": row[6],
                    "school": row[7],
                    "timeframe_start": row[8],
                    "timeframe_end": row[9],
                    "description": row[10],
                    # Add compatibility fields
                    "technique": row[3],
                    "timeframe": f"{row[8]}-{row[9]}" if row[8] and row[9] else row[5],
                    "relevance_score": 0.5,  # Default score for keyword matches
                }
                results.append(artwork)

            return results

        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []

    def _comprehensive_nlp_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Comprehensive search with NLP metadata extraction."""
        if not self.nlp:
            print("spaCy model not available, falling back to semantic search")
            return self._semantic_only_search(query, k)

        try:
            # Correct typos in the query using TextBlob
            corrected_query = str(TextBlob(query.strip()).correct())
            # Note: We have the spaCy model available but don't need to process the doc for basic keyword extraction

            # Extract metadata fields
            metadata_fields = {}

            # Heuristic keyword lists for art metadata
            type_keywords = [
                "genre",
                "historical",
                "interior",
                "landscape",
                "mythological",
                "other",
                "portrait",
                "religious",
                "still-life",
                "nude",
            ]

            school_keywords = [
                "flemish",
                "italian",
                "dutch",
                "french",
                "german",
                "spanish",
                "english",
                "american",
                "austrian",
                "russian",
                "belgian",
            ]

            technique_keywords = [
                "oil",
                "canvas",
                "panel",
                "wood",
                "fresco",
                "tempera",
                "watercolor",
                "pastel",
                "drawing",
                "sketch",
                "engraving",
                "medium",  # Add the actual column name
            ]

            # Extract entities and keywords
            query_lower = corrected_query.lower()

            # Check for type keywords
            for keyword in type_keywords:
                if keyword in query_lower:
                    metadata_fields["type"] = keyword
                    break

            # Check for school keywords
            for keyword in school_keywords:
                if keyword in query_lower:
                    metadata_fields["school"] = keyword
                    break

            # Check for technique keywords
            for keyword in technique_keywords:
                if keyword in query_lower:
                    metadata_fields["technique"] = keyword
                    break

            # Perform filtered search based on extracted metadata
            if metadata_fields:
                return self._filtered_search_with_metadata(
                    corrected_query, metadata_fields, k
                )
            else:
                return self._semantic_only_search(corrected_query, k)

        except Exception as e:
            print(f"Error in comprehensive NLP search: {e}")
            return self._semantic_only_search(query, k)

    def _filtered_search_with_metadata(
        self, query: str, metadata: Dict[str, str], k: int
    ) -> List[Dict[str, Any]]:
        """Search with metadata filters."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Build dynamic SQL query based on metadata
            where_conditions = []
            params = []

            for field, value in metadata.items():
                # Map field names to actual database columns
                if field == "technique":
                    field = "medium"
                where_conditions.append(f"{field} LIKE ?")
                params.append(f"%{value}%")

            where_clause = " AND ".join(where_conditions)

            cursor.execute(
                f"""
                SELECT image_file, author, title, medium, dimensions, date, type, school, 
                       timeframe_start, timeframe_end, full_description
                FROM artworks 
                WHERE {where_clause}
                LIMIT ?
            """,
                params + [k * 2],
            )  # Get more results for semantic filtering

            filtered_artworks = []
            for row in cursor.fetchall():
                filtered_artworks.append(
                    {
                        "id": row[0],  # image_file as id
                        "image_file": row[0],
                        "author": row[1],
                        "title": row[2],
                        "medium": row[3],
                        "dimensions": row[4],
                        "date": row[5],
                        "type": row[6],
                        "school": row[7],
                        "timeframe_start": row[8],
                        "timeframe_end": row[9],
                        "description": row[10],
                        # Add compatibility fields
                        "technique": row[3],
                        "timeframe": (
                            f"{row[8]}-{row[9]}" if row[8] and row[9] else row[5]
                        ),
                    }
                )

            # If we have filtered results, perform semantic ranking on them
            if filtered_artworks:
                # Score each artwork based on semantic similarity
                for artwork in filtered_artworks:
                    # Create a text representation for similarity scoring
                    artwork_text = f"{artwork['title']} {artwork['author']} {artwork['description']}"

                    # Get embedding similarity (simplified)
                    query_embedding = self.model.encode([query])
                    artwork_embedding = self.model.encode([artwork_text])

                    # Calculate cosine similarity
                    import numpy as np

                    similarity = np.dot(query_embedding[0], artwork_embedding[0]) / (
                        np.linalg.norm(query_embedding[0])
                        * np.linalg.norm(artwork_embedding[0])
                    )
                    artwork["relevance_score"] = max(0.0, similarity)

                # Sort by relevance score
                filtered_artworks.sort(key=lambda x: x["relevance_score"], reverse=True)
                return filtered_artworks[:k]
            else:
                # If no filtered results, fall back to semantic search
                return self._semantic_only_search(query, k)

        except Exception as e:
            print(f"Error in filtered search: {e}")
            return self._semantic_only_search(query, k)

    def _hybrid_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword approaches."""
        # Get results from both approaches
        semantic_results = self._semantic_only_search(query, k)
        keyword_results = self._keyword_only_search(query, k)

        # Combine and deduplicate
        seen_ids = set()
        combined_results = []

        # Add semantic results first (higher priority)
        for result in semantic_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                combined_results.append(result)

        # Add keyword results that aren't already included
        for result in keyword_results:
            if result["id"] not in seen_ids and len(combined_results) < k:
                seen_ids.add(result["id"])
                combined_results.append(result)

        return combined_results[:k]

    def close(self):
        """Close database connections."""
        if hasattr(self._thread_local, "conn"):
            self._thread_local.conn.close()

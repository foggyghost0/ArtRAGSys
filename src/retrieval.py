"""
Retrieval module for Art RAG System using ChromaDB.
Provides semantic and hybrid search capabilities.
"""

import sqlite3
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class ArtSearch:
    """Art search system with ChromaDB vector search and SQLite metadata."""

    def __init__(
        self, db_path: str = "art_database.db", chroma_path: str = "./chroma_db"
    ):
        """Initialize the search system."""
        print("Initializing art search system...")

        # Load sentence transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to SQLite database
        self.conn = sqlite3.connect(db_path)

        # Connect to ChromaDB
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
                        "similarity": 1 - distance,  # Convert distance to similarity
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

    def text_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform full-text search using SQLite FTS."""
        try:
            cursor = self.conn.execute(
                """
            SELECT s.id, s.sentence_text, s.sentence_order, 
                   a.image_file, a.title, a.author, a.type, a.school
            FROM sentences_fts f
            JOIN description_sentences s ON f.rowid = s.id
            JOIN artworks a ON s.artwork_id = a.image_file
            WHERE f.sentence_text MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
                (query, k),
            )

            results = cursor.fetchall()
            formatted_results = []

            for result in results:
                formatted_results.append(
                    {
                        "sentence_id": result[0],
                        "sentence": result[1],
                        "sentence_order": result[2],
                        "artwork_id": result[3],
                        "title": result[4],
                        "author": result[5],
                        "type": result[6],
                        "school": result[7],
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in text search: {e}")
            return []

    def metadata_search(
        self,
        author: Optional[str] = None,
        art_type: Optional[str] = None,
        school: Optional[str] = None,
        timeframe_start: Optional[int] = None,
        timeframe_end: Optional[int] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search artworks by metadata criteria."""
        try:
            conditions = []
            params = []

            if author:
                conditions.append("a.author LIKE ?")
                params.append(f"%{author}%")

            if art_type:
                conditions.append("a.type LIKE ?")
                params.append(f"%{art_type}%")

            if school:
                conditions.append("a.school LIKE ?")
                params.append(f"%{school}%")

            if timeframe_start:
                conditions.append("a.timeframe_start >= ?")
                params.append(timeframe_start)

            if timeframe_end:
                conditions.append("a.timeframe_end <= ?")
                params.append(timeframe_end)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(k)

            query = f"""
            SELECT a.image_file, a.title, a.author, a.type, a.school, 
                   a.date, a.timeframe_start, a.timeframe_end, a.full_description
            FROM artworks a
            WHERE {where_clause}
            LIMIT ?
            """

            cursor = self.conn.execute(query, params)
            results = cursor.fetchall()

            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "artwork_id": result[0],
                        "title": result[1],
                        "author": result[2],
                        "type": result[3],
                        "school": result[4],
                        "date": result[5],
                        "timeframe_start": result[6],
                        "timeframe_end": result[7],
                        "description": result[8],
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in metadata search: {e}")
            return []

    def hybrid_search(
        self, query: str, k: int = 5, semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Combine semantic and text search results."""
        try:
            # Get semantic results
            semantic_results = self.semantic_search_sentences(query, k * 2)

            # Get text search results
            text_results = self.text_search(query, k * 2)

            # Combine and deduplicate by sentence ID
            combined_results = {}

            # Add semantic results with weighted scores
            for result in semantic_results:
                sentence_key = (
                    result["artwork_id"] + "_" + str(result["sentence_order"])
                )
                result["score"] = result["similarity"] * semantic_weight
                result["search_type"] = "semantic"
                combined_results[sentence_key] = result

            # Add text results with weighted scores
            for result in text_results:
                sentence_key = (
                    result["artwork_id"] + "_" + str(result["sentence_order"])
                )
                text_score = 1.0 - semantic_weight  # Simple scoring for text matches

                if sentence_key in combined_results:
                    # Boost score if found in both searches
                    combined_results[sentence_key]["score"] += text_score
                    combined_results[sentence_key]["search_type"] = "hybrid"
                else:
                    result["score"] = text_score
                    result["search_type"] = "text"
                    result["similarity"] = text_score
                    combined_results[sentence_key] = result

            # Sort by combined score and return top k
            sorted_results = sorted(
                combined_results.values(), key=lambda x: x["score"], reverse=True
            )

            return sorted_results[:k]

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def close(self):
        """Close database connections."""
        if self.conn:
            self.conn.close()


def demo_search(query: str = "royal portrait painting"):
    """Demonstrate different search capabilities."""
    print(f"Demo search for: '{query}'")
    print("=" * 50)

    # Initialize search system
    search = ArtSearch()

    # Semantic sentence search
    print("\n1. Semantic Sentence Search:")
    print("-" * 30)
    semantic_results = search.semantic_search_sentences(query, k=3)
    for i, result in enumerate(semantic_results, 1):
        print(
            f"{i}. [{result['similarity']:.3f}] {result['title']} by {result['author']}"
        )
        print(f"   {result['sentence'][:100]}...")
        print()

    # Semantic description search
    print("\n2. Semantic Description Search:")
    print("-" * 30)
    desc_results = search.semantic_search_descriptions(query, k=3)
    for i, result in enumerate(desc_results, 1):
        print(
            f"{i}. [{result['similarity']:.3f}] {result['title']} by {result['author']}"
        )
        print(f"   {result['description'][:100]}...")
        print()

    # Text search
    print("\n3. Full-Text Search:")
    print("-" * 30)
    text_results = search.text_search(query, k=3)
    for i, result in enumerate(text_results, 1):
        print(f"{i}. {result['title']} by {result['author']}")
        print(f"   {result['sentence'][:100]}...")
        print()

    # Hybrid search
    print("\n4. Hybrid Search:")
    print("-" * 30)
    hybrid_results = search.hybrid_search(query, k=3)
    for i, result in enumerate(hybrid_results, 1):
        print(
            f"{i}. [{result['score']:.3f}] {result['title']} by {result['author']} ({result['search_type']})"
        )
        print(f"   {result.get('sentence', result.get('description', ''))[:100]}...")
        print()

    # Metadata search
    print("\n5. Metadata Search (portraits):")
    print("-" * 30)
    metadata_results = search.metadata_search(art_type="portrait", k=3)
    for i, result in enumerate(metadata_results, 1):
        print(f"{i}. {result['title']} by {result['author']} ({result['date']})")
        print(f"   {result['description'][:100]}...")
        print()

    search.close()


if __name__ == "__main__":
    demo_search("royal portrait painting")

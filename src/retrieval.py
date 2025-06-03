"""
Retrieval module for Art RAG System using ChromaDB.
Provides semantic and hybrid search capabilities.
"""

import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

import spacy

from textblob import TextBlob


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
        """Search artworks by metadata criteria with relevance ranking."""
        try:
            conditions = []
            params = []
            relevance_parts = []

            # Build SQL conditions and relevance scoring
            if author:
                conditions.append("a.author LIKE ?")
                params.append(f"%{author}%")
                # Add relevance boost for exact author matches
                relevance_parts.append(
                    f"CASE WHEN a.author LIKE '%{author}%' THEN 1.0 ELSE 0.0 END"
                )

            if art_type:
                conditions.append("a.type LIKE ?")
                params.append(f"%{art_type}%")
                relevance_parts.append(
                    f"CASE WHEN a.type LIKE '%{art_type}%' THEN 1.0 ELSE 0.0 END"
                )

            if school:
                conditions.append("a.school LIKE ?")
                params.append(f"%{school}%")
                relevance_parts.append(
                    f"CASE WHEN a.school LIKE '%{school}%' THEN 1.0 ELSE 0.0 END"
                )

            if timeframe_start:
                conditions.append("a.timeframe_start >= ?")
                params.append(timeframe_start)
                relevance_parts.append("0.5")  # Time match gets lower weight

            if timeframe_end:
                conditions.append("a.timeframe_end <= ?")
                params.append(timeframe_end)
                relevance_parts.append("0.5")  # Time match gets lower weight

            # Build relevance score calculation
            if relevance_parts:
                relevance_calc = " + ".join(relevance_parts)
            else:
                relevance_calc = "1.0"  # Default relevance if no specific criteria

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(k)

            query = f"""
            SELECT a.image_file, a.title, a.author, a.type, a.school, 
                   a.date, a.timeframe_start, a.timeframe_end, a.full_description,
                   ({relevance_calc}) / {len(relevance_parts) if relevance_parts else 1} as relevance_score
            FROM artworks a
            WHERE {where_clause}
            ORDER BY relevance_score DESC
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
                        "relevance_score": result[9],
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in metadata search: {e}")
            return []

    def _normalize_result_key(self, result: Dict[str, Any]) -> str:
        """Generate a unique key for deduplication across different search methods."""
        artwork_id = result.get("artwork_id", "")
        sentence_order = result.get("sentence_order")

        if sentence_order is not None:
            return f"{artwork_id}#{sentence_order}"
        else:
            return artwork_id

    def _normalize_result_format(
        self, result: Dict[str, Any], search_type: str
    ) -> Dict[str, Any]:
        """Normalize result format across different search methods."""
        normalized = {
            "artwork_id": result.get("artwork_id", ""),
            "title": result.get("title", ""),
            "author": result.get("author", ""),
            "type": result.get("type", ""),
            "school": result.get("school", ""),
            "search_type": search_type,
            "content": "",
            "relevance_score": 0.0,
        }

        # Extract content and relevance score based on search type
        if search_type in ["semantic_sentences", "text"]:
            normalized["content"] = result.get("sentence", "")
            normalized["sentence_order"] = result.get("sentence_order")
            normalized["relevance_score"] = result.get(
                "similarity", result.get("score", 0.0)
            )
        elif search_type == "semantic_descriptions":
            normalized["content"] = result.get("description", "")
            normalized["relevance_score"] = result.get("similarity", 0.0)
        elif search_type == "metadata":
            normalized["content"] = result.get("description", "")
            normalized["relevance_score"] = result.get("relevance_score", 1.0)
        else:
            # For hybrid or other types
            normalized["content"] = result.get(
                "sentence", result.get("description", "")
            )
            normalized["relevance_score"] = result.get(
                "score", result.get("similarity", 0.0)
            )
            if "sentence_order" in result:
                normalized["sentence_order"] = result["sentence_order"]

        return normalized

    def reciprocal_rank_fusion(
        self, result_lists: List[Tuple[List[Dict[str, Any]], str]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine multiple ranked result lists.

        Args:
            result_lists: List of (results, search_type) tuples
            k: RRF parameter (typically 60)

        Returns:
            List of results ranked by RRF score
        """
        rrf_scores = defaultdict(float)
        result_lookup = {}

        for results, search_type in result_lists:
            for rank, result in enumerate(results):
                # Normalize the result format
                normalized_result = self._normalize_result_format(result, search_type)
                result_key = self._normalize_result_key(normalized_result)

                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
                rrf_scores[result_key] += rrf_contribution

                # Store the result (prefer sentence-level over artwork-level)
                if (
                    result_key not in result_lookup
                    or "sentence_order" in normalized_result
                ):
                    result_lookup[result_key] = normalized_result

        # Add RRF scores to results and sort
        final_results = []
        for result_key, rrf_score in rrf_scores.items():
            result = result_lookup[result_key]
            result["rrf_score"] = rrf_score
            final_results.append(result)

        # Sort by RRF score (descending)
        final_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return final_results

    def comprehensive_search(
        self,
        query: str,
        k: int = 10,
        use_semantic_sentences: bool = True,
        use_semantic_descriptions: bool = True,
        use_text_search: bool = True,
        use_metadata_search: bool = True,
        metadata_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive search using RRF to combine all available search methods.

        Args:
            query: Search query
            k: Number of final results to return
            use_semantic_sentences: Whether to include semantic sentence search
            use_semantic_descriptions: Whether to include semantic description search
            use_text_search: Whether to include full-text search
            use_metadata_search: Whether to include metadata search
            metadata_params: Additional metadata search parameters

        Returns:
            List of results ranked by RRF score
        """
        try:
            result_lists = []
            search_k = k * 3  # Get more results from each method for better fusion

            # Semantic sentence search
            if use_semantic_sentences:
                semantic_sent_results = self.semantic_search_sentences(query, search_k)
                if semantic_sent_results:
                    result_lists.append((semantic_sent_results, "semantic_sentences"))

            # Semantic description search
            if use_semantic_descriptions:
                semantic_desc_results = self.semantic_search_descriptions(
                    query, search_k
                )
                if semantic_desc_results:
                    result_lists.append(
                        (semantic_desc_results, "semantic_descriptions")
                    )

            # Text search
            if use_text_search:
                text_results = self.text_search(query, search_k)
                if text_results:
                    result_lists.append((text_results, "text"))

            # Metadata search (if parameters provided or can be extracted from query)
            if use_metadata_search:
                # Use provided metadata params or try to extract from query
                if metadata_params:
                    meta_results = self.metadata_search(k=search_k, **metadata_params)
                else:
                    # Try to extract metadata from query using preprocess_query
                    query_analysis = preprocess_query(query)
                    if query_analysis["metadata_fields"]:
                        meta_results = self.metadata_search(
                            k=search_k, **query_analysis["metadata_fields"]
                        )
                    else:
                        meta_results = []

                if meta_results:
                    result_lists.append((meta_results, "metadata"))

            # Apply RRF if we have multiple result lists
            if len(result_lists) > 1:
                fused_results = self.reciprocal_rank_fusion(result_lists)
                return fused_results[:k]
            elif len(result_lists) == 1:
                # Only one search method used, just format and return
                results, search_type = result_lists[0]
                normalized_results = [
                    self._normalize_result_format(result, search_type)
                    for result in results[:k]
                ]
                # Add RRF score as the original relevance score
                for result in normalized_results:
                    result["rrf_score"] = result["relevance_score"]
                return normalized_results
            else:
                return []

        except Exception as e:
            print(f"Error in comprehensive search: {e}")
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


def demo_comprehensive_search(query: str = "royal portrait painting"):
    """Demonstrate comprehensive search with RRF fusion."""
    print(f"Comprehensive RRF Search Demo for: '{query}'")
    print("=" * 60)

    # Initialize search system
    search = ArtSearch()

    # Run comprehensive search
    print("\nComprehensive Search with RRF:")
    print("-" * 40)
    comprehensive_results = search.comprehensive_search(query, k=5)

    for i, result in enumerate(comprehensive_results, 1):
        print(
            f"{i}. [RRF: {result['rrf_score']:.4f}] {result['title']} by {result['author']}"
        )
        print(f"   Search Type: {result['search_type']}")
        print(f"   Content: {result['content'][:100]}...")
        if "sentence_order" in result:
            print(f"   Sentence #{result['sentence_order']}")
        print()

    # Compare with individual methods
    print("\nComparison with Individual Methods:")
    print("-" * 40)

    # Semantic sentence search
    print("\nSemantic Sentences (top 3):")
    semantic_results = search.semantic_search_sentences(query, k=3)
    for i, result in enumerate(semantic_results, 1):
        print(f"  {i}. [{result['similarity']:.3f}] {result['title']}")

    # Text search
    print("\nText Search (top 3):")
    text_results = search.text_search(query, k=3)
    for i, result in enumerate(text_results, 1):
        print(f"  {i}. {result['title']}")

    # Metadata search for portraits
    print("\nMetadata Search - portraits (top 3):")
    metadata_results = search.metadata_search(art_type="portrait", k=3)
    for i, result in enumerate(metadata_results, 1):
        print(f"  {i}. [{result['relevance_score']:.3f}] {result['title']}")

    search.close()


def preprocess_query(query: str) -> dict:
    """
    Analyze the query using spaCy transformer model and heuristics.
    Decide which search types to perform and extract relevant snippets.
    Returns:
        {
            "semantic": bool,
            "text": bool,
            "hybrid": bool,
            "metadata": bool,
            "semantic_snippet": str,
            "text_snippet": str,
            "hybrid_snippet": str,
            "metadata_fields": dict  # e.g. {"author": "Rembrandt", "type": "portrait"}
        }
    """

    # Correct typos in the query using TextBlob
    query = str(TextBlob(query.strip()).correct())
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(query)

    # Metadata fields to extract
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
        "study",
    ]
    school_keywords = [
        "American",
        "Austrian",
        "Belgian",
        "Bohemian",
        "Catalan",
        "Danish",
        "Dutch",
        "English",
        "Flemish",
        "French",
        "German",
        "Greek",
        "Hungarian",
        "Irish",
        "Italian",
        "Netherlandish",
        "Norwegian",
        "Other",
        "Polish",
        "Portuguese",
        "Russian",
        "Scottish",
        "Spanish",
        "Swedish",
        "Swiss",
    ]
    # Lowercase for matching
    type_keywords_lc = [kw.lower() for kw in type_keywords]
    school_keywords_lc = [kw.lower() for kw in school_keywords]

    # 1. Named Entity Recognition for author, date, school
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            metadata_fields["author"] = ent.text
        elif ent.label_ == "DATE":
            # Try to extract years or centuries
            year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", ent.text)
            century_match = re.search(
                r"\b(\d{1,2})(st|nd|rd|th)? century\b", ent.text.lower()
            )
            if year_match:
                year = int(year_match.group(1))
                # Heuristic: treat as either start or end
                if "start" not in metadata_fields:
                    metadata_fields["timeframe_start"] = year
                else:
                    metadata_fields["timeframe_end"] = year
            elif century_match:
                century = int(century_match.group(1))
                # Approximate century to years
                start = (century - 1) * 100 + 1
                end = century * 100
                metadata_fields["timeframe_start"] = start
                metadata_fields["timeframe_end"] = end
        elif ent.label_ == "ORG":
            # Possible school
            metadata_fields["school"] = ent.text

    # 2. Keyword matching for type and school
    for token in doc:
        token_lc = token.text.lower()
        if token_lc in type_keywords_lc:
            metadata_fields["type"] = token.text
        if token_lc in school_keywords_lc:
            metadata_fields["school"] = token.text

    # 3. Decide which searches to perform
    has_metadata = bool(metadata_fields)
    # If query contains both metadata and non-metadata (descriptive) terms, do hybrid
    non_metadata_tokens = [
        token.text
        for token in doc
        if not token.is_stop
        and token.pos_ in {"NOUN", "ADJ", "VERB"}
        and token.text.lower() not in type_keywords_lc + school_keywords_lc
    ]
    # If query is only metadata, prefer metadata search
    only_metadata = len(non_metadata_tokens) == 0 and has_metadata

    # Always do metadata search if any metadata field found
    do_metadata = has_metadata
    # Do hybrid if both metadata and descriptive tokens
    do_hybrid = has_metadata and len(non_metadata_tokens) > 0
    # Do semantic/text if query is descriptive or hybrid
    do_semantic = not only_metadata or do_hybrid
    do_text = do_semantic  # For now, treat text and semantic similarly

    # Extract snippets
    semantic_snippet = " ".join(non_metadata_tokens) if non_metadata_tokens else query
    text_snippet = semantic_snippet
    hybrid_snippet = query

    return {
        "semantic": do_semantic,
        "text": do_text,
        "hybrid": do_hybrid,
        "metadata": do_metadata,
        "semantic_snippet": semantic_snippet,
        "text_snippet": text_snippet,
        "hybrid_snippet": hybrid_snippet,
        "metadata_fields": metadata_fields,
    }


if __name__ == "__main__":
    # Test the new comprehensive search functionality
    print("Testing Art Search System with RRF")
    print("=" * 50)

    # Test queries
    test_queries = [
        "royal portrait painting",
        "landscape with trees",
        "Dutch still life",
        "religious painting by Italian artist",
    ]

    for query in test_queries:
        print(f"\n\nTesting query: '{query}'")
        print("-" * 50)
        demo_comprehensive_search(query)
        print("\n" + "=" * 50)

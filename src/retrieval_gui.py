"""
Thread-safe retrieval module for Art RAG System GUI.
Provides semantic and hybrid search capabilities for GUI applications.
"""

import sqlite3
import threading
import math
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

import chromadb
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
from rapidfuzz import fuzz


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
            print(f"- Full Descriptions: {self.description_collection.count()} items")

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
                        "id": metadata["artwork_id"],  # Add id field for consistency
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
                        "id": metadata["artwork_id"],  # Add id field for consistency
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

    def _semantic_only_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Semantic search only with improved scoring."""
        sentence_results = self.semantic_search_sentences(query, k * 2)
        description_results = self.semantic_search_descriptions(query, k * 2)

        # Combine and deduplicate with weighted scoring
        artwork_scores = defaultdict(
            lambda: {"sentence_scores": [], "description_scores": []}
        )

        for result in sentence_results:
            artwork_scores[result["artwork_id"]]["sentence_scores"].append(
                result["similarity"]
            )

        for result in description_results:
            artwork_scores[result["artwork_id"]]["description_scores"].append(
                result["similarity"]
            )

        # Calculate weighted scores (descriptions weighted higher than individual sentences)
        final_scores = []
        for artwork_id, scores in artwork_scores.items():
            sentence_avg = (
                sum(scores["sentence_scores"]) / len(scores["sentence_scores"])
                if scores["sentence_scores"]
                else 0.0
            )
            description_avg = (
                sum(scores["description_scores"]) / len(scores["description_scores"])
                if scores["description_scores"]
                else 0.0
            )

            # Weight descriptions higher than sentences
            combined_score = 0.3 * sentence_avg + 0.7 * description_avg
            final_scores.append((artwork_id, combined_score))

        # Sort by score and get top k
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Get full artwork details
        results = []
        for artwork_id, score in final_scores[:k]:
            artwork = self.get_artwork_by_id(artwork_id)
            if artwork:
                artwork["relevance_score"] = score
                results.append(artwork)

        # Normalize scores
        return self._normalize_scores(results, "relevance_score")

    def _comprehensive_nlp_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Comprehensive search with NLP metadata extraction."""
        if not self.nlp:
            print("spaCy model not available, falling back to semantic search")
            return self._semantic_only_search(query, k)

        try:
            # Correct typos in the query using TextBlob
            corrected_query = str(TextBlob(query.strip()).correct())

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

                    # Get embedding similarity
                    query_embedding = self.model.encode([query])
                    artwork_embedding = self.model.encode([artwork_text])

                    # Calculate cosine similarity
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

    def close(self):
        """Close database connections."""
        if hasattr(self._thread_local, "conn"):
            self._thread_local.conn.close()

    def _calculate_corpus_stats(self) -> Dict[str, Any]:
        """Calculate corpus statistics for BM25."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM artworks")
            doc_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT full_description FROM artworks WHERE full_description IS NOT NULL"
            )
            descriptions = cursor.fetchall()

            total_doc_length = 0
            term_doc_freq = defaultdict(int)

            for (desc,) in descriptions:
                if desc:
                    words = desc.lower().split()
                    total_doc_length += len(words)
                    unique_words = set(words)
                    for word in unique_words:
                        term_doc_freq[word] += 1

            avg_doc_length = total_doc_length / max(len(descriptions), 1)

            return {
                "doc_count": doc_count,
                "avg_doc_length": avg_doc_length,
                "term_doc_freq": dict(term_doc_freq),
            }
        except Exception as e:
            print(f"Error calculating corpus stats: {e}")
            return {"doc_count": 1, "avg_doc_length": 100, "term_doc_freq": {}}

    def _bm25_score(
        self,
        query_terms: List[str],
        document_fields: Dict[str, str],
        field_weights: Dict[str, float] = None,
        corpus_stats: Dict[str, Any] = None,
    ) -> float:
        """Calculate BM25 score for a document given query terms."""
        if field_weights is None:
            field_weights = {
                "title": 3.0,
                "author": 2.0,
                "description": 1.0,
                "type": 1.5,
                "school": 1.5,
                "medium": 1.2,
            }

        if corpus_stats is None:
            corpus_stats = self._calculate_corpus_stats()

        k1, b = 1.2, 0.75  # BM25 parameters
        total_score = 0.0

        for field, text in document_fields.items():
            if not text or field not in field_weights:
                continue

            field_weight = field_weights[field]
            doc_terms = text.lower().split()
            doc_len = len(doc_terms)

            if doc_len == 0:
                continue

            term_freq = Counter(doc_terms)
            field_score = 0.0

            for term in query_terms:
                term_lower = term.lower()
                tf = term_freq.get(term_lower, 0)

                if tf > 0:
                    # Calculate IDF
                    df = corpus_stats["term_doc_freq"].get(term_lower, 1)
                    idf = math.log((corpus_stats["doc_count"] - df + 0.5) / (df + 0.5))

                    # Calculate BM25 component
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (
                        1 - b + b * doc_len / corpus_stats["avg_doc_length"]
                    )

                    field_score += idf * (numerator / denominator)

            total_score += field_score * field_weight

        return max(0.0, total_score)

    def _normalize_scores(
        self, results: List[Dict[str, Any]], score_field: str = "relevance_score"
    ) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not results:
            return results

        scores = [r.get(score_field, 0.0) for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same, set them all to 1.0
            for result in results:
                result[f"{score_field}_normalized"] = 1.0
        else:
            # Min-max normalization
            for result in results:
                original_score = result.get(score_field, 0.0)
                normalized = (original_score - min_score) / (max_score - min_score)
                result[f"{score_field}_normalized"] = normalized

        return results

    def _reciprocal_rank_fusion(
        self, ranked_lists: List[List[Dict[str, Any]]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion."""
        rrf_scores = defaultdict(float)
        all_items = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, 1):
                item_id = item["id"]
                rrf_scores[item_id] += 1.0 / (k + rank)
                all_items[item_id] = item

        # Sort by RRF score and normalize
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for item_id, score in sorted_items:
            item = all_items[item_id].copy()
            item["rrf_score"] = score
            item["relevance_score"] = score  # For consistency
            results.append(item)

        # Normalize RRF scores
        return self._normalize_scores(results, "rrf_score")

    def _enhanced_keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Enhanced keyword search with BM25 ranking."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Get corpus statistics for BM25
            corpus_stats = self._calculate_corpus_stats()

            # Get all artworks for BM25 scoring
            cursor.execute(
                """
                SELECT image_file, author, title, medium, dimensions, date, type, school, 
                       timeframe_start, timeframe_end, full_description
                FROM artworks
            """
            )

            query_terms = query.lower().split()
            scored_results = []

            for row in cursor.fetchall():
                document_fields = {
                    "title": str(row[2] or ""),
                    "author": str(row[1] or ""),
                    "description": str(row[10] or ""),
                    "type": str(row[6] or ""),
                    "school": str(row[7] or ""),
                    "medium": str(row[3] or ""),
                }

                # Calculate BM25 score
                bm25_score = self._bm25_score(
                    query_terms, document_fields, corpus_stats=corpus_stats
                )

                if bm25_score > 0:
                    artwork = {
                        "id": row[0],
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
                        "technique": row[3],
                        "timeframe": (
                            f"{row[8]}-{row[9]}" if row[8] and row[9] else row[5]
                        ),
                        "relevance_score": bm25_score,
                        "bm25_score": bm25_score,
                    }
                    scored_results.append(artwork)

            # Sort by BM25 score and normalize
            scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
            top_results = scored_results[:k]

            return self._normalize_scores(top_results, "bm25_score")

        except Exception as e:
            print(f"Error in enhanced keyword search: {e}")
            return []

    def advanced_hybrid_retrieval(
        self, query: str, k: int = 12
    ) -> List[Dict[str, Any]]:
        """Advanced hybrid retrieval using RRF."""
        try:
            # Get results from different methods
            semantic_results = self.semantic_search_descriptions(query, k * 2)
            keyword_results = self._enhanced_keyword_search(query, k * 2)
            fuzzy_results = self.fuzzy_keyword_retrieval(query, k * 2)

            # Prepare ranked lists for RRF
            ranked_lists = []

            if semantic_results:
                # Normalize semantic scores
                semantic_normalized = self._normalize_scores(
                    semantic_results, "similarity"
                )
                ranked_lists.append(semantic_normalized)

            if keyword_results:
                ranked_lists.append(
                    keyword_results
                )  # Already normalized in _enhanced_keyword_search

            if fuzzy_results:
                # Normalize fuzzy scores
                fuzzy_normalized = self._normalize_scores(
                    fuzzy_results, "relevance_score"
                )
                ranked_lists.append(fuzzy_normalized)

            if not ranked_lists:
                return []

            # Combine using RRF
            rrf_results = self._reciprocal_rank_fusion(ranked_lists, k=60)

            # Get full artwork details for top results
            final_results = []
            for result in rrf_results[:k]:
                artwork = self.get_artwork_by_id(result["id"])
                if artwork:
                    artwork.update(
                        {
                            "rrf_score": result.get("rrf_score", 0.0),
                            "rrf_score_normalized": result.get(
                                "rrf_score_normalized", 0.0
                            ),
                            "relevance_score": result.get("rrf_score_normalized", 0.0),
                        }
                    )
                    final_results.append(artwork)

            return final_results

        except Exception as e:
            print(f"Error in advanced hybrid retrieval: {e}")
            return []

    def enhanced_comprehensive_search(
        self, query: str, search_type: str = "advanced_hybrid", k: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Enhanced comprehensive search with improved ranking methods.

        Args:
            query: Search query
            search_type: Type of search ("advanced_hybrid", "bm25", "rrf_only", "semantic", "fuzzy", "metadata")
            k: Number of results to return
        """
        try:
            # Map legacy search types to new methods
            if search_type in [
                "all",
                "comprehensive",
                "keyword",
                "text",
                "hybrid_scoring",
            ]:
                search_type = "advanced_hybrid"

            if search_type == "advanced_hybrid":
                return self.advanced_hybrid_retrieval(query, k)
            elif search_type == "bm25":
                return self._enhanced_keyword_search(query, k)
            elif search_type == "rrf_only":
                # RRF of semantic and BM25 keyword search
                semantic_results = self.semantic_search_descriptions(query, k * 2)
                keyword_results = self._enhanced_keyword_search(query, k * 2)
                ranked_lists = (
                    [semantic_results, keyword_results]
                    if semantic_results and keyword_results
                    else []
                )
                if ranked_lists:
                    rrf_results = self._reciprocal_rank_fusion(ranked_lists)
                    return rrf_results[:k]
                return []
            elif search_type == "semantic":
                return self._semantic_only_search(query, k)
            elif search_type == "fuzzy":
                return self.fuzzy_keyword_retrieval(query, k)
            elif search_type == "metadata":
                return self._comprehensive_nlp_search(query, k)
            else:
                # Default to advanced hybrid for unknown types
                return self.advanced_hybrid_retrieval(query, k)

        except Exception as e:
            print(f"Error in enhanced comprehensive search: {e}")
            return []

    def fuzzy_keyword_retrieval(
        self, query: str, k: int = 5, threshold: int = 70
    ) -> List[Dict[str, Any]]:
        """Fuzzy keyword-based retrieval using rapidfuzz for approximate matches."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT image_file, author, title, medium, dimensions, date, type, school, 
                       timeframe_start, timeframe_end, full_description
                FROM artworks
                """
            )
            all_rows = cursor.fetchall()
            scored_results = []
            for row in all_rows:
                fields = [
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[10] or ""),
                    str(row[6] or ""),
                    str(row[7] or ""),
                    str(row[3] or ""),
                ]
                max_score = max(
                    [
                        fuzz.partial_ratio(query.lower(), field.lower())
                        for field in fields
                        if field
                    ]
                )
                if max_score >= threshold:
                    scored_results.append((max_score, row))
            # Sort by fuzzy score descending
            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, row in scored_results[:k]:
                artwork = {
                    "id": row[0],
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
                    "technique": row[3],
                    "timeframe": f"{row[8]}-{row[9]}" if row[8] and row[9] else row[5],
                    "relevance_score": score / 100.0,
                }
                results.append(artwork)
            return results
        except Exception as e:
            print(f"Error in fuzzy keyword search: {e}")
            return []

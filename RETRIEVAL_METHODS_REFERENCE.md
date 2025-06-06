# ArtRAG Retrieval Methods Quick Reference

This document summarizes the main retrieval methods available in the ArtRAG System, describing their differences and use cases.

---

## Core Retrieval Methods

## 1. `semantic_retrieval`
- **Description:** Uses sentence-transformer embeddings and ChromaDB to find artworks semantically similar to the query.
- **Strength:** Captures meaning and context, not just keywords.
- **Use case:** When you want results that are conceptually related, even if the exact words don't match.

## 2. `keyword_retrieval` *Consider BM25 instead*
- **Description:** Performs a standard SQL `LIKE` keyword search across artwork metadata fields.
- **Strength:** Fast, simple, and effective for exact or partial word matches.
- **Use case:** When you want to find artworks containing specific words or phrases.
- **Note:** Consider using `enhanced_comprehensive_search(query, "bm25")` for better keyword ranking.

## 3. `fuzzy_keyword_retrieval`
- **Description:** Uses fuzzy string matching (via rapidfuzz) to find artworks with metadata fields similar to the query, even if there are typos or slight differences.
- **Strength:** Tolerant to misspellings and variations.
- **Use case:** When users may make typos or use slightly different terms.

## 4. `hybrid_scoring_retrieval` 
- **Description:** Calculates a weighted score for each artwork by combining semantic similarity and fuzzy keyword match scores.
- **Strength:** Provides a nuanced ranking that leverages both meaning and textual similarity.
- **Use case:** When you want the most relevant results, considering both semantics and fuzzy keyword matches.

## Advanced Ranking Methods (RECOMMENDED)

## 5. `advanced_hybrid_retrieval` **BEST**
- **Description:** Uses Reciprocal Rank Fusion (RRF) to combine semantic, BM25-enhanced keyword, and fuzzy search results with sophisticated ranking.
- **Strength:** Combines multiple ranking signals using state-of-the-art fusion techniques; produces high-quality results.
- **Use case:** When you want the most advanced ranking that combines multiple search strategies optimally.

## 6. `enhanced_comprehensive_search` **RECOMMENDED**
- **Description:** Entry point for all advanced ranking methods including BM25, RRF, and advanced hybrid approaches.
- **Search Types:**
  - `"advanced_hybrid"`: Uses RRF to combine semantic, BM25, and fuzzy search **BEST**
  - `"bm25"`: Uses BM25 algorithm for keyword ranking instead of simple LIKE queries
  - `"rrf_only"`: Uses RRF to combine semantic and traditional keyword search
- **Strength:** Provides access to cutting-edge information retrieval algorithms.
- **Use case:** When you want the best possible search results using modern ranking algorithms.

---

## Advanced Features

### BM25 Scoring
- **Field Weights:** Title (3.0), Author (2.0), Type (1.5), School (1.5), Medium (1.2), Description (1.0)
- **Algorithm:** Uses TF-IDF with document length normalization
- **Advantage:** Much better than simple LIKE queries for keyword relevance

### Reciprocal Rank Fusion (RRF)
- **Algorithm:** Combines multiple ranked lists using reciprocal rank scores
- **Parameters:** k=60 (fusion parameter)
- **Advantage:** Statistically proven to outperform simple score averaging

### Score Normalization
- **Method:** Min-max normalization to 0-1 range
- **Application:** Applied before combining different search methods
- **Advantage:** Ensures fair combination of scores from different algorithms

---

## GUI Integration

The modern GUI (`gui_modern.py`) now includes dropdown options for:
- **advanced_hybrid** - NEW: Advanced hybrid with RRF (DEFAULT)
- **bm25** - NEW: BM25-enhanced keyword search  
- **rrf_only** - NEW: RRF combination of semantic + keyword
- **semantic** - Semantic search only
- **fuzzy** - Fuzzy keyword search
- **hybrid_scoring** - Weighted score combination
- **comprehensive** - Now uses enhanced comprehensive search with better algorithms
- **text** - Basic keyword search (consider BM25 instead)
- **metadata** - Now uses enhanced metadata-aware search

---

**Recommendation:**
- Use **`advanced_hybrid`** for the best overall search experience
- Use **`bm25`** for improved keyword-based search
- Use **`rrf_only`** for optimal combination of semantic and keyword results
- Use **`semantic_retrieval`** for meaning-based queries
- Use **`fuzzy_keyword_retrieval`** for typo-tolerant search
- Use **`hybrid_scoring_retrieval`** for balanced semantic + fuzzy scoring

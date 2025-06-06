# ArtRAG Retrieval Methods Quick Reference

This document summarizes the main retrieval methods available in the ArtRAG System, describing their differences and use cases.

---

## Core Retrieval Methods

## 1. `semantic_search_sentences` & `semantic_search_descriptions`
- **Description:** Uses sentence-transformer embeddings and ChromaDB to find artworks semantically similar to the query.
- **Strength:** Captures meaning and context, not just keywords.
- **Use case:** When you want results that are conceptually related, even if the exact words don't match.

## 2. `fuzzy_keyword_retrieval`
- **Description:** Uses fuzzy string matching (via rapidfuzz) to find artworks with metadata fields similar to the query, even if there are typos or slight differences.
- **Strength:** Tolerant to misspellings and variations.
- **Use case:** When users may make typos or use slightly different terms.

## Advanced Ranking Methods (RECOMMENDED)

## 3. `advanced_hybrid_retrieval` **BEST**
- **Description:** Uses Reciprocal Rank Fusion (RRF) to combine semantic, BM25-enhanced keyword, and fuzzy search results with sophisticated ranking.
- **Strength:** Combines multiple ranking signals; produces high-quality results.
- **Use case:** When you want the most advanced ranking that combines multiple search strategies optimally.

## 4. `enhanced_comprehensive_search` **RECOMMENDED**
- **Description:** Entry point for all advanced ranking methods including BM25, RRF, and advanced hybrid approaches.
- **Search Types:**
  - `"advanced_hybrid"`: Uses RRF to combine semantic, BM25, and fuzzy search **BEST**
  - `"bm25"`: Uses BM25 algorithm for keyword ranking instead of simple LIKE queries
  - `"rrf_only"`: Uses RRF to combine semantic and BM25 keyword search
  - `"semantic"`: Semantic search only with improved scoring
  - `"fuzzy"`: Fuzzy keyword search
  - `"metadata"`: NLP-enhanced metadata extraction and search
- **Strength:** Provides access to multiple information retrieval algorithms.
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

The modern GUI (`gui_modern.py`) includes dropdown options for:
- **advanced_hybrid** - Advanced hybrid with RRF (DEFAULT) **BEST**
- **bm25** - BM25-enhanced keyword search  
- **rrf_only** - RRF combination of semantic + BM25 keyword
- **semantic** - Semantic search only
- **fuzzy** - Fuzzy keyword search
- **metadata** - Enhanced metadata-aware search with NLP NER

---

**Recommendation:**
- Use **`advanced_hybrid`** for the best overall search experience
- Use **`bm25`** for improved keyword-based search
- Use **`rrf_only`** for optimal combination of semantic and keyword results
- Use **`semantic`** for meaning-based queries
- Use **`fuzzy`** for typo-tolerant search
- Use **`metadata`** for queries that might contain art-specific metadata

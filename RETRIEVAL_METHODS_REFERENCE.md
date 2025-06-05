# ArtRAG Retrieval Methods Quick Reference

This document summarizes the main retrieval methods available in the ArtRAG System, describing their differences and use cases.

---

## 1. `semantic_retrieval`
- **Description:** Uses sentence-transformer embeddings and ChromaDB to find artworks semantically similar to the query.
- **Strength:** Captures meaning and context, not just keywords.
- **Use case:** When you want results that are conceptually related, even if the exact words don't match.

## 2. `keyword_retrieval`
- **Description:** Performs a standard SQL `LIKE` keyword search across artwork metadata fields.
- **Strength:** Fast, simple, and effective for exact or partial word matches.
- **Use case:** When you want to find artworks containing specific words or phrases.

## 3. `fuzzy_keyword_retrieval`
- **Description:** Uses fuzzy string matching (via rapidfuzz) to find artworks with metadata fields similar to the query, even if there are typos or slight differences.
- **Strength:** Tolerant to misspellings and variations.
- **Use case:** When users may make typos or use slightly different terms.

## 4. `hybrid_retrieval`
- **Description:** Combines results from both semantic and keyword retrieval, deduplicates, and returns the top results.
- **Strength:** Balances semantic and keyword matches, but does not combine scores.
- **Use case:** When you want a mix of both approaches, prioritizing semantic results.

## 5. `hybrid_scoring_retrieval`
- **Description:** Calculates a weighted score for each artwork by combining semantic similarity and fuzzy keyword match scores.
- **Strength:** Provides a nuanced ranking that leverages both meaning and textual similarity.
- **Use case:** When you want the most relevant results, considering both semantics and fuzzy keyword matches.

## 6. `comprehensive_search`
- **Description:** A flexible entry point that can use semantic, keyword, or NLP-enhanced (metadata) search depending on the `search_type` argument.
- **Strength:** Adapts to the search type requested; can use advanced NLP for metadata extraction.
- **Use case:** When you want to let the system choose the best strategy or use metadata-aware search.

---

**Tip:**
- Use `semantic_retrieval` for meaning-based queries.
- Use `keyword_retrieval` for exact word/phrase matches.
- Use `fuzzy_keyword_retrieval` for typo-tolerant search.
- Use `hybrid_scoring_retrieval` for the best overall ranking.
- Use `comprehensive_search` for advanced or metadata-driven queries.

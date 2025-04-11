from init_databases import ArtSearch

def main_search(text_query, semantic_query):
    """
    Perform semantic and hybrid search on the art database."""
    print("Initializing art search...")
    
    search = ArtSearch()
    
    # Example semantic search (using vector embeddings)
    query = semantic_query
    print(f"Performing semantic search for: {query}")
    semantic_results = search.semantic_search(query)
    
    if not semantic_results:
        print("No semantic search results found.")
    else:
        print(f"Found {len(semantic_results)} semantic search results:")
        for i, r in enumerate(semantic_results, 1):
            print(f"{i}. {r['title']} by {r['author']}: {r['sentence']}")
    
    # Example hybrid search (combining text and semantic search)
    print("\nPerforming hybrid search...")
    text_query = query  # Keywords for text search
    semantic_query = semantic_query  # Semantic understanding
    hybrid_results = search.hybrid_search(text_query, semantic_query)
    
    if not hybrid_results:
        print("No hybrid search results found.")
    else:
        print(f"Found {len(hybrid_results)} hybrid search results:")
        for i, result in enumerate(hybrid_results, 1):
            # Hybrid search returns tuples of (id, sentence)
            print(f"{i}. ID: {result[0]}, Text: {result[1]}")

if __name__ == '__main__':
    main_search("flower", "still life with flowers")
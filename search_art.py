from retrieval import ArtSearch

def main():
    print("Initializing art search...")
    search = ArtSearch()
    
    # Example semantic search (using vector embeddings)
    query = "royal portrait"
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
    text_query = "portrait"  # Keywords for text search
    semantic_query = "royal portrait"  # Semantic understanding
    hybrid_results = search.hybrid_search(text_query, semantic_query)
    
    if not hybrid_results:
        print("No hybrid search results found.")
    else:
        print(f"Found {len(hybrid_results)} hybrid search results:")
        for i, result in enumerate(hybrid_results, 1):
            # Hybrid search returns tuples of (id, sentence)
            print(f"{i}. ID: {result[0]}, Text: {result[1]}")

if __name__ == '__main__':
    main()
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sqlite3
import hnswlib
import numpy as np
from init_knowledge_graph import Namespace 

KG_FILE = "knowledge_graph.ttl"
DB_FILE = "artrag.db"
EMBEDDING_MODEL_FILE = "entity_embeddings.npy"
HNSW_INDEX_FILE = "hnsw_index.bin"

def hybrid_retrieve(query):
    """
    Retrieves information from both SQLite, the knowledge graph, and HNSW index.
    """
    # Determine query type (simple keyword search or relationship-based)
    if is_relationship_query(query):
        kg_results = retrieve_from_kg(query)
        sqlite_results = []
    else:
        kg_results = []
        sqlite_results = retrieve_from_sqlite(query)

    # Enhance retrieval with HNSW similarity search
    hnsw_results = retrieve_similar_artworks(query) # Added HNSW retrieval

    # Combine and rank results
    combined_results = combine_results(sqlite_results, kg_results, hnsw_results, query) # Added HNSW results
    return combined_results

def is_relationship_query(query):
    """
    Simple heuristic to determine if the query is relationship-based.
    """
    keywords = ["related", "influenced", "similar", "connection"]
    return any(keyword in query.lower() for keyword in keywords)

def retrieve_from_kg(query):
    """
    Retrieves triples from the knowledge graph based on a SPARQL query.
    """
    kg = Graph()
    try:
        kg.parse(KG_FILE, format="turtle")
    except Exception as e:
        print(f"Error parsing KG file: {e}")
        return []

    sparql_query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
            FILTER (
                regex(str(?subject), "%s", "i") ||
                regex(str(?predicate), "%s", "i") ||
                regex(str(?object), "%s", "i")
            )
        }
    """ % (query, query, query)

    try:
        results = kg.query(sparql_query)
        return [(str(row.subject), str(row.predicate), str(row.object)) for row in results]
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return []

def retrieve_from_sqlite(query):
    """
    Retrieves artworks from SQLite based on a full-text search.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        # Query the artwork_search virtual table correctly
        cursor.execute(
            """
            SELECT artworks.id, artworks.image_file, artworks.title, artworks.description, 
                   artworks.type_id, artworks.timeframe 
            FROM artwork_search
            JOIN artworks ON artwork_search.rowid = artworks.id
            WHERE artwork_search MATCH ?
            """,
            (query,)
        )
        results = cursor.fetchall()
        return results
    except sqlite3.Error as e:
        print(f"SQLite query error: {e}")
        return []
    finally:
        conn.close()

def retrieve_similar_artworks(query, top_k=5):
    """
    Retrieves similar artworks using HNSW index based on a text query.
    """
    try:
        # Load entity embeddings and HNSW index
        entity_embeddings = np.load(EMBEDDING_MODEL_FILE)
        hnsw_index = hnswlib.Index(space='cosine', dim=entity_embeddings.shape[1])
        hnsw_index.load_index(HNSW_INDEX_FILE)
    except Exception as e:
        print(f"Error loading embeddings or index: {e}")
        return []

    # Load SQLite database to find matching artwork
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Try to find direct matches in titles (case insensitive)
    cursor.execute("SELECT id, title FROM artworks WHERE LOWER(title) LIKE LOWER(?)", (f"%{query}%",))
    artwork_matches = cursor.fetchall()
    
    # If no matches in title, check authors too
    if not artwork_matches:
        cursor.execute("""
            SELECT artworks.id, artworks.title 
            FROM artworks 
            JOIN authors ON artworks.author_id = authors.id
            WHERE LOWER(authors.name) LIKE LOWER(?)
        """, (f"%{query}%",))
        artwork_matches = cursor.fetchall()
    
    # If still no matches, just return empty list
    if not artwork_matches:
        print(f"No artwork or author matching '{query}' found.")
        conn.close()
        return []
    
    # Use the first match
    artwork_id, matched_title = artwork_matches[0]
    print(f"Found match: '{matched_title}' (ID: {artwork_id})")

    # Get the index of the artwork in the embeddings array
    artwork_index = artwork_id - 1  # Assuming artwork IDs start from 1

    # Check if the artwork index is valid
    if artwork_index < 0 or artwork_index >= len(entity_embeddings):
        print(f"Artwork index {artwork_index} is out of bounds.")
        conn.close()
        return []

    # Query HNSW index for similar artworks
    labels, distances = hnsw_index.knn_query(entity_embeddings[artwork_index].reshape(1, -1), k=top_k+1)

    # Fetch all artwork titles for mapping
    cursor.execute("SELECT id, title FROM artworks")
    all_artwork_titles = {id: title for id, title in cursor.fetchall()}
    conn.close()

    # Map the labels back to artwork titles (excluding the query artwork itself)
    similar_artworks = []
    for label in labels[0]:
        similar_id = label + 1  # Assuming artwork IDs start from 1
        if similar_id in all_artwork_titles and similar_id != artwork_id:
            similar_artworks.append(all_artwork_titles[similar_id])
    
    return similar_artworks

def combine_results(sqlite_results, kg_results, hnsw_results, query):
    """
    Combines results from SQLite, the knowledge graph, and HNSW using Reciprocal Rank Fusion.
    """
    # Constant k for RRF formula (typical value is 60)
    k = 60
    
    # Create dictionaries to store rankings from each source
    all_items = {}
    
    # Process SQLite results
    for rank, (artwork_id, image_file, title, description, type_id, timeframe) in enumerate(sqlite_results):
        item_id = f"artwork:{artwork_id}"
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'SQLite',
                'image_file': image_file,
                'title': title,
                'description': description,
                'rrf_score': 0
            }
        # Add RRF score contribution from SQLite ranking
        all_items[item_id]['rrf_score'] += 1 / (k + rank)
    
    # Process KG results (convert triples to a form that can be ranked)
    for rank, (subject, predicate, obj) in enumerate(kg_results):
        item_id = subject  # Use subject as identifier
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'Knowledge Graph',
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'rrf_score': 0
            }
        # Add RRF score contribution from KG ranking
        all_items[item_id]['rrf_score'] += 1 / (k + rank)
    
    # Process HNSW results
    for rank, artwork_title in enumerate(hnsw_results):
        # Use title as identifier (ideally we'd use artwork_id, but we only have titles)
        item_id = f"title:{artwork_title}"
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'HNSW Similarity',
                'title': artwork_title,
                'rrf_score': 0
            }
        # Add RRF score contribution from HNSW ranking
        all_items[item_id]['rrf_score'] += 1 / (k + rank)
    
    # Convert dictionary to list and sort by RRF score
    ranked_results = list(all_items.values())
    ranked_results.sort(key=lambda x: x['rrf_score'], reverse=True)
    
    return ranked_results

# Example usage
if __name__ == "__main__":
    query = "flowers"
    # Example query to test the hybrid retrieval
    print(f"Retrieving results for query: '{query}'")
    results = hybrid_retrieve(query)
    if results:
        print(results[0])
    else:
        print("No results found.")
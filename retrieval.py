from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sqlite3
#from init_knowledge_graph import namespaces #removed
import hnswlib
import numpy as np
from init_knowledge_graph import Namespace #added

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
    combined_results = combine_results(sqlite_results, kg_results, hnsw_results) # Added HNSW results
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
        cursor.execute(
            """
            SELECT image_file, title, description FROM artworks
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
    # Load entity embeddings and HNSW index
    entity_embeddings = np.load(EMBEDDING_MODEL_FILE)
    hnsw_index = hnswlib.Index(space='cosine', dim=entity_embeddings.shape[1])
    hnsw_index.load_index(HNSW_INDEX_FILE)

    # Load SQLite database to map artwork titles to IDs
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM artworks")
    artwork_titles = cursor.fetchall()
    conn.close()

    # Find the artwork ID that matches the query
    artwork_id = None
    for id, title in artwork_titles:
        if query.lower() in title.lower():
            artwork_id = id
            break

    if artwork_id is None:
        print(f"Artwork with title '{query}' not found.")
        return []

    # Get the index of the artwork in the embeddings array
    artwork_index = artwork_id - 1  # Assuming artwork IDs start from 1

    # Check if the artwork index is valid
    if artwork_index < 0 or artwork_index >= len(entity_embeddings):
        print(f"Artwork index {artwork_index} is out of bounds.")
        return []

    # Query HNSW index for similar artworks
    labels, distances = hnsw_index.knn_query(entity_embeddings[artwork_index].reshape(1, -1), k=top_k)

     # Load SQLite database to map artwork titles to IDs
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM artworks")
    artwork_titles = cursor.fetchall()
    conn.close()

    # Map the labels back to artwork titles
    similar_artworks = []
    for label in labels[0]:
        # Find the artwork ID that matches the label
        artwork_id = label + 1  # Assuming artwork IDs start from 1
        for id, title in artwork_titles:
            if id == artwork_id:
                similar_artworks.append(title)
                break

    return similar_artworks

def combine_results(sqlite_results, kg_results, hnsw_results):
    """
    Combines and ranks results from SQLite, the knowledge graph, and HNSW.
    """
    # Simple concatenation for demonstration
    combined = sqlite_results + kg_results + [(artwork, "HNSW Similarity") for artwork in hnsw_results]
    return combined
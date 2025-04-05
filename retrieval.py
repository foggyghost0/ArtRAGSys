from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sqlite3
from init_knowledge_graph import namespaces
KG_FILE = "knowledge_graph.ttl"
DB_FILE = "artrag.db"

def hybrid_retrieve(query):
    """
    Retrieves information from both SQLite and the knowledge graph.
    """
    # Determine query type (simple keyword search or relationship-based)
    if is_relationship_query(query):  # Implement this function
        kg_results = retrieve_from_kg(query)
        sqlite_results = []  # No SQLite query for relationship queries
    else:
        kg_results = []  # No KG query for simple keyword queries
        sqlite_results = retrieve_from_sqlite(query)

    # Combine and rank results (implement this function)
    combined_results = combine_results(sqlite_results, kg_results)
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

def combine_results(sqlite_results, kg_results):
    """
    Combines and ranks results from SQLite and the knowledge graph.
    This is a placeholder; implement your ranking logic here.
    """
    # Simple concatenation for demonstration
    combined = sqlite_results + kg_results
    return combined
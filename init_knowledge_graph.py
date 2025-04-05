import sqlite3
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

# Database file path
DB_FILE = 'artrag.db'
KG_FILE = "knowledge_graph.ttl"

# Define namespaces
ART = Namespace("ART")  # Namespace for artworks
AUTH = Namespace("AUTH")  # Namespace for authors
SCHOOL = Namespace("SCHOOL")  # Namespace for schools
TYPE = Namespace("TYPE")  # Namespace for artwork types
REL = Namespace("REL")  # Namespace for relationships

def create_knowledge_graph():
    """
    Creates a knowledge graph from the SQLite database.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create an RDF Graph
    kg = Graph()

    # Bind namespaces
    kg.bind("art", ART)
    kg.bind("auth", AUTH)
    kg.bind("school", SCHOOL)
    kg.bind("type", TYPE)
    kg.bind("rel", REL)

    # 1. Add Authors
    cursor.execute("SELECT id, name FROM authors")
    authors = cursor.fetchall()
    for author_id, author_name in authors:
        author_uri = AUTH[str(author_id)]
        kg.add((author_uri, RDF.type, REL.Author))  # Define as an Author type
        kg.add((author_uri, RDFS.label, Literal(author_name)))  # Add the name

    # 2. Add Schools
    cursor.execute("SELECT id, name FROM schools")
    schools = cursor.fetchall()
    for school_id, school_name in schools:
        school_uri = SCHOOL[str(school_id)]
        kg.add((school_uri, RDF.type, REL.School))  # Define as a School type
        kg.add((school_uri, RDFS.label, Literal(school_name)))  # Add the name

    # 3. Add Artwork Types
    cursor.execute("SELECT id, name FROM artwork_types")
    artwork_types = cursor.fetchall()
    for type_id, type_name in artwork_types:
        type_uri = TYPE[str(type_id)]
        kg.add((type_uri, RDF.type, REL.ArtworkType))  # Define as an ArtworkType
        kg.add((type_uri, RDFS.label, Literal(type_name)))  # Add the name

    # 4. Add Artworks and their relationships
    cursor.execute("""
        SELECT 
            artworks.id, artworks.image_file, artworks.title, 
            artworks.author_id, artworks.type_id, artworks.school_id
        FROM artworks
    """)
    artworks = cursor.fetchall()
    for artwork_id, image_file, title, author_id, type_id, school_id in artworks:
        artwork_uri = ART[str(artwork_id)]
        kg.add((artwork_uri, RDF.type, REL.Artwork))  # Define as an Artwork
        kg.add((artwork_uri, RDFS.label, Literal(title)))  # Add the title
        kg.add((artwork_uri, REL.image_file, Literal(image_file)))  # Link to image file

        # Link to Author
        if author_id:
            author_uri = AUTH[str(author_id)]
            kg.add((artwork_uri, REL.created_by, author_uri))

        # Link to Artwork Type
        if type_id:
            type_uri = TYPE[str(type_id)]
            kg.add((artwork_uri, REL.is_a, type_uri))

        # Link to School
        if school_id:
            school_uri = SCHOOL[str(school_id)]
            kg.add((artwork_uri, REL.belongs_to, school_uri))

    # Serialize the knowledge graph to a file
    try:
        kg.serialize(KG_FILE, format="turtle")
        print(f"Knowledge graph created and saved to {KG_FILE}")
    except Exception as e:
        print(f"Error serializing knowledge graph: {e}")

    conn.close()

if __name__ == '__main__':
    create_knowledge_graph()
import sqlite3
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import numpy as np
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import hnswlib
import os

# Database file path
DB_FILE = 'artrag.db'
KG_FILE = "knowledge_graph.ttl"
EMBEDDING_MODEL_FILE = "entity_embeddings.npy"
HNSW_INDEX_FILE = "hnsw_index.bin"

# Define namespaces
ART = Namespace("http://example.org/art/")
AUTH = Namespace("http://example.org/author/")
SCHOOL = Namespace("http://example.org/school/")
TYPE = Namespace("http://example.org/type/")
REL = Namespace("http://example.org/relation/")

def create_knowledge_graph():
    """
    Creates a knowledge graph from the SQLite database and generates embeddings.
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
        kg.add((author_uri, RDF.type, REL.Author))
        kg.add((author_uri, RDFS.label, Literal(author_name)))

    # 2. Add Schools
    cursor.execute("SELECT id, name FROM schools")
    schools = cursor.fetchall()
    for school_id, school_name in schools:
        school_uri = SCHOOL[str(school_id)]
        kg.add((school_uri, RDF.type, REL.School))
        kg.add((school_uri, RDFS.label, Literal(school_name)))

    # 3. Add Artwork Types
    cursor.execute("SELECT id, name FROM artwork_types")
    artwork_types = cursor.fetchall()
    for type_id, type_name in artwork_types:
        type_uri = TYPE[str(type_id)]
        kg.add((type_uri, RDF.type, REL.ArtworkType))
        kg.add((type_uri, RDFS.label, Literal(type_name)))

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
        kg.add((artwork_uri, RDF.type, REL.Artwork))
        kg.add((artwork_uri, RDFS.label, Literal(title)))
        kg.add((artwork_uri, REL.image_file, Literal(image_file)))

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

    # Generate Knowledge Graph Embeddings
    generate_embeddings(kg)


def generate_embeddings(kg):
    """
    Generates knowledge graph embeddings using PyKEEN and builds HNSW index.
    """
    # Convert RDF graph to PyKEEN TriplesFactory
    triples = []
    for s, p, o in kg:
        triples.append((str(s), str(p), str(o)))

    triples_factory = TriplesFactory.from_labeled_triples(np.array(triples))

    # Split the triples factory into training and testing
    training_factory = triples_factory
    testing_factory = triples_factory  # Using same for both as we're not evaluating performance

    # Define and train the TransE model using PyKEEN pipeline
    result = pipeline(
        model='TransE',
        training=training_factory,
        testing=testing_factory,  # Provide both training and testing
        random_seed=123,
        training_kwargs={'num_epochs': 50}
    )

    # Fix: Access entity embeddings directly
    entity_embeddings = result.model.entity_representations[0]().detach().numpy()
    # Save entity embeddings
    np.save(EMBEDDING_MODEL_FILE, entity_embeddings)
    print(f"Entity embeddings saved to {EMBEDDING_MODEL_FILE}")

    # Build HNSW index
    build_hnsw_index(entity_embeddings, triples_factory.entity_to_id)

def build_hnsw_index(embeddings, entity_to_id):
    """
    Builds HNSW index for fast similarity search.
    """
    # Initialize HNSW index
    num_elements = len(embeddings)
    embedding_size = embeddings.shape[1]
    hnsw_index = hnswlib.Index(space='cosine', dim=embedding_size)  # You can also use 'l2' for Euclidean distance

    # Initialize index
    hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)  # Adjust parameters as needed

    # Add data to index
    hnsw_index.add_items(embeddings, np.arange(num_elements))  # Use indices as labels

    # Set query time accuracy
    hnsw_index.set_ef(50)  # Adjust as needed

    # Save index
    hnsw_index.save_index(HNSW_INDEX_FILE)
    print(f"HNSW index saved to {HNSW_INDEX_FILE}")

if __name__ == '__main__':
    create_knowledge_graph()
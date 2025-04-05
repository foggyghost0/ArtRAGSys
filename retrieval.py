from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sqlite3
import hnswlib
import numpy as np
import re
import spacy
from init_knowledge_graph import Namespace 

# Load SpaCy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

KG_FILE = "knowledge_graph.ttl"
DB_FILE = "artrag.db"
EMBEDDING_MODEL_FILE = "entity_embeddings.npy"
HNSW_INDEX_FILE = "hnsw_index.bin"

def hybrid_retrieve(query):
    """
    Enhanced retrieval system that combines NER, SQLite, knowledge graph, and HNSW index.
    """
    # Process the query with NER to extract entities
    query_entities = extract_entities(query)
    print(f"Extracted entities: {query_entities}")
    
    # Determine if this is a relationship query
    relationship_score = calculate_relationship_score(query)
    print(f"Relationship score: {relationship_score}")
    is_relation = relationship_score > 0.6  # Threshold for considering it a relationship query
    
    # Always search in SQLite with weights based on entity types
    sqlite_results = retrieve_from_sqlite_enhanced(query, query_entities)
    print(f"SQLite results: {len(sqlite_results)}")
    
    # Search for sentences that match the query
    sentence_results = search_in_sentences(query)
    print(f"Sentence results: {len(sentence_results)}")
    
    # Always get similar artworks using HNSW index
    hnsw_results = retrieve_similar_artworks(query)
    print(f"HNSW results: {len(hnsw_results)}")
    
    # Get knowledge graph results if it's a relationship query
    kg_results = retrieve_from_kg(query) if is_relation else []
    print(f"Knowledge graph results: {len(kg_results)}")
    
    # Combine all initial results
    combined_results = combine_weighted_results(
        sqlite_results, 
        sentence_results,
        hnsw_results, 
        kg_results, 
        query,
        query_entities,
        relationship_score
    )
    print(f"Combined results: {len(combined_results)}")
    
    # Fallback: If no results and we have PERSON entities, try direct author search
    if not combined_results and query_entities["PERSON"]:
        print("No initial results, trying direct author search...")
        for person in query_entities["PERSON"]:
            author_results = direct_author_search(person)
            if author_results:
                combined_results.extend(author_results)
    
    return combined_results

def extract_entities(query):
    """
    Extract named entities from the query using SpaCy.
    Returns a dictionary of entities by type.
    """
    doc = nlp(query)
    entities = {
        "PERSON": [],
        "DATE": [],
        "ORG": [],
        "WORK_OF_ART": [],
        "GPE": [],  # Countries, cities, states
        "GENERAL": []  # For queries without specific entities
    }
    
    # Extract entities from SpaCy's NER
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities["GENERAL"].append(ent.text)
    
    # If no entities found, put the whole query in GENERAL
    if all(not entities[key] for key in entities):
        entities["GENERAL"] = [query]
    
    # Additional pattern matching for dates
    date_patterns = [
        r'\b(1[0-9]{3}|20[0-2][0-9])\b',  # Years 1000-2029
        r'\b(1[0-9]|20)th century\b',     # Century references
        r'\b([0-9]{1,2})s\b'              # Decades
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if match not in entities["DATE"]:
                entities["DATE"].append(match)
    
    return entities

def calculate_relationship_score(query):
    """
    Calculate a score indicating how likely the query is about relationships.
    Returns a score between 0 and 1.
    """
    relationship_terms = [
        "related", "connection", "influenced", "inspired", "teacher", 
        "student", "contemporary", "similar to", "compared", "versus",
        "vs", "relationship", "between", "and", "connection", "linked",
        "associated", "compared to", "same school", "movement"
    ]
    
    # Normalize query
    query_lower = query.lower()
    
    # Calculate score based on presence of terms
    score = 0
    for term in relationship_terms:
        if term in query_lower:
            score += 1
    
    # Normalize score to 0-1 range
    return min(score / 3, 1.0)  # 3+ matches give a score of 1

def retrieve_from_sqlite_enhanced(query, query_entities):
    """
    Enhanced SQLite retrieval that uses entity information to search specific tables.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    results = []
    
    try:
        # 1. Search for authors if PERSON entities are present
        if query_entities["PERSON"]:
            for person in query_entities["PERSON"]:
                cursor.execute("""
                    SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                           'author_match' as match_type, auth.name as author_name
                    FROM artworks a
                    JOIN authors auth ON a.author_id = auth.id
                    WHERE LOWER(auth.name) LIKE LOWER(?)
                    LIMIT 10
                """, (f"%{person}%",))
                author_results = cursor.fetchall()
                for row in author_results:
                    results.append(dict(row))
        
        # 2. Search for dates if DATE entities are present
        if query_entities["DATE"]:
            for date in query_entities["DATE"]:
                cursor.execute("""
                    SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                           'date_match' as match_type
                    FROM artworks a
                    WHERE LOWER(a.date) LIKE LOWER(?) OR LOWER(a.timeframe) LIKE LOWER(?)
                    LIMIT 10
                """, (f"%{date}%", f"%{date}%"))
                date_results = cursor.fetchall()
                for row in date_results:
                    results.append(dict(row))
        
        # 3. Search for schools if ORG entities are present
        if query_entities["ORG"] or "school" in query.lower():
            org_terms = query_entities["ORG"] + (["school"] if "school" in query.lower() else [])
            for org in org_terms:
                cursor.execute("""
                    SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                           'school_match' as match_type, s.name as school_name
                    FROM artworks a
                    JOIN schools s ON a.school_id = s.id
                    WHERE LOWER(s.name) LIKE LOWER(?)
                    LIMIT 10
                """, (f"%{org}%",))
                school_results = cursor.fetchall()
                for row in school_results:
                    results.append(dict(row))
        
        # 4. Search for artwork types explicitly mentioned
        if "painting" in query.lower() or "sculpture" in query.lower() or "drawing" in query.lower():
            art_types = []
            if "painting" in query.lower(): art_types.append("painting")
            if "sculpture" in query.lower(): art_types.append("sculpture")
            if "drawing" in query.lower(): art_types.append("drawing")
            
            for art_type in art_types:
                cursor.execute("""
                    SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                           'type_match' as match_type, t.name as type_name
                    FROM artworks a
                    JOIN artwork_types t ON a.type_id = t.id
                    WHERE LOWER(t.name) LIKE LOWER(?)
                    LIMIT 10
                """, (f"%{art_type}%",))
                type_results = cursor.fetchall()
                for row in type_results:
                    results.append(dict(row))
        
        # 5. Always perform general search using FTS for any query terms
        cursor.execute("""
            SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                   'title_technique_match' as match_type
            FROM artwork_search
            JOIN artworks a ON artwork_search.rowid = a.id
            WHERE artwork_search MATCH ?
            LIMIT 20
        """, (query,))
        general_results = cursor.fetchall()
        for row in general_results:
            results.append(dict(row))
        
        # Remove duplicates based on artwork id
        unique_results = {}
        for row in results:
            if row['id'] not in unique_results:
                unique_results[row['id']] = row
        
        return list(unique_results.values())
    
    except sqlite3.Error as e:
        print(f"SQLite query error: {e}")
        return []
    finally:
        conn.close()

def search_in_sentences(query):
    """
    Search for sentences that match the query using FTS5.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT s.id, s.artwork_id, s.sentence, a.title, a.image_file
            FROM sentence_search
            JOIN description_sentences s ON sentence_search.rowid = s.id
            JOIN artworks a ON s.artwork_id = a.id
            WHERE sentence_search MATCH ?
            ORDER BY rank
            LIMIT 20
        """, (query,))
        
        results = [dict(row) for row in cursor.fetchall()]
        return results
    
    except sqlite3.Error as e:
        print(f"Error searching sentences: {e}")
        return []
    finally:
        conn.close()

def retrieve_from_kg(query):
    """
    Enhanced knowledge graph retrieval using SPARQL.
    """
    kg = Graph()
    try:
        kg.parse(KG_FILE, format="turtle")
    except Exception as e:
        print(f"Error parsing KG file: {e}")
        return []
    
    # Enhanced SPARQL query to find relationships
    sparql_query = """
        SELECT ?subject ?predicate ?object ?s_label ?p_label ?o_label
        WHERE {
            ?subject ?predicate ?object .
            OPTIONAL { ?subject rdfs:label ?s_label . }
            OPTIONAL { ?predicate rdfs:label ?p_label . }
            OPTIONAL { ?object rdfs:label ?o_label . }
            FILTER (
                regex(str(?subject), "%s", "i") ||
                regex(str(?predicate), "%s", "i") ||
                regex(str(?object), "%s", "i") ||
                regex(str(?s_label), "%s", "i") ||
                regex(str(?o_label), "%s", "i")
            )
        }
    """ % (query, query, query, query, query)

    try:
        results = kg.query(sparql_query)
        formatted_results = []
        
        for row in results:
            subject = str(row.subject)
            predicate = str(row.predicate)
            obj = str(row.object)
            
            # Use labels if available
            s_label = str(row.s_label) if row.s_label else subject
            p_label = str(row.p_label) if row.p_label else predicate.split('/')[-1]
            o_label = str(row.o_label) if row.o_label else obj
            
            formatted_results.append({
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'subject_label': s_label,
                'predicate_label': p_label,
                'object_label': o_label,
                'source': 'Knowledge Graph'
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return []

def direct_author_search(author_name):
    """
    Direct search for an author by name with multiple case variations.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    results = []
    
    # Try different case variations
    name_variations = [
        author_name,                     # Original
        author_name.lower(),             # lowercase
        author_name.upper(),             # UPPERCASE
        author_name.title(),             # Title Case
        author_name.capitalize(),        # Capitalized
        # Try with just part of the name (for last names)
        author_name.split()[-1] if ' ' in author_name else author_name
    ]
    
    try:
        for name_var in name_variations:
            # Direct search in authors table
            cursor.execute("""
                SELECT a.id, a.image_file, a.title, a.description, a.type_id, a.timeframe,
                       'author_match' as match_type, auth.name as author_name
                FROM artworks a
                JOIN authors auth ON a.author_id = auth.id
                WHERE auth.name LIKE ?
                LIMIT 10
            """, (f"%{name_var}%",))
            
            author_results = cursor.fetchall()
            for row in author_results:
                results.append(dict(row))
            
            # If we found results, no need to try other variations
            if results:
                print(f"Found match with name variation: '{name_var}'")
                break
                
        # Remove duplicates
        unique_results = {}
        for row in results:
            if row['id'] not in unique_results:
                unique_results[row['id']] = row
                
        return list(unique_results.values())
        
    except sqlite3.Error as e:
        print(f"SQLite direct author search error: {e}")
        return []
    finally:
        conn.close()

def retrieve_similar_artworks(query, top_k=5):
    """
    Enhanced HNSW similarity search that returns more complete information.
    """
    try:
        # Load entity embeddings and HNSW index
        entity_embeddings = np.load(EMBEDDING_MODEL_FILE)
        hnsw_index = hnswlib.Index(space='cosine', dim=entity_embeddings.shape[1])
        hnsw_index.load_index(HNSW_INDEX_FILE)
    except Exception as e:
        print(f"Error loading embeddings or index: {e}")
        return []

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Try to find direct matches in titles, authors, and descriptions
    cursor.execute("""
        SELECT a.id, a.title, a.image_file, a.description 
        FROM artwork_search 
        JOIN artworks a ON artwork_search.rowid = a.id
        WHERE artwork_search MATCH ?
        ORDER BY rank
        LIMIT 1
    """, (query,))
    
    artwork_match = cursor.fetchone()
    
    # If no match, try authors
    if not artwork_match:
        cursor.execute("""
            SELECT a.id, a.title, a.image_file, a.description
            FROM artworks a
            JOIN authors auth ON a.author_id = auth.id
            WHERE LOWER(auth.name) LIKE LOWER(?)
            LIMIT 1
        """, (f"%{query}%",))
        artwork_match = cursor.fetchone()
    
    # If still no match, just return empty list
    if not artwork_match:
        conn.close()
        return []
    
    artwork_id = artwork_match['id']
    artwork_index = artwork_id - 1  # Assuming artwork IDs start from 1
    
    # Check if the artwork index is valid
    if artwork_index < 0 or artwork_index >= len(entity_embeddings):
        conn.close()
        return []
    
    # Query HNSW index for similar artworks
    labels, distances = hnsw_index.knn_query(entity_embeddings[artwork_index].reshape(1, -1), k=top_k+1)
    
    # Get complete artwork information for the similar items
    similar_artworks = []
    for label in labels[0]:
        similar_id = label + 1  # Assuming artwork IDs start from 1
        if similar_id != artwork_id:  # Don't include the query artwork
            cursor.execute("""
                SELECT a.id, a.title, a.image_file, a.description, a.timeframe,
                       auth.name as author_name, t.name as type_name
                FROM artworks a
                LEFT JOIN authors auth ON a.author_id = auth.id
                LEFT JOIN artwork_types t ON a.type_id = t.id
                WHERE a.id = ?
            """, (similar_id,))
            
            similar_artwork = cursor.fetchone()
            if similar_artwork:
                result = dict(similar_artwork)
                result['source'] = 'HNSW Similarity'
                similar_artworks.append(result)
    
    conn.close()
    return similar_artworks

def combine_weighted_results(sqlite_results, sentence_results, hnsw_results, kg_results, query, query_entities, relationship_score):
    """
    Combine all results using weighted Reciprocal Rank Fusion.
    """
    # Constants for RRF and weighting
    k = 60  # RRF constant
    
    # Adjust weights based on query characteristics
    weights = {
        'sqlite': 1.0,
        'sentences': 0.8,
        'hnsw': 1.5 if not query_entities["PERSON"] else 0.8,
        'kg': 1.0 if relationship_score > 0.5 else 0.3
    }
    
    # Boost certain match types
    match_type_boosts = {
        'author_match': 1.5 if query_entities["PERSON"] else 1.0,
        'date_match': 1.5 if query_entities["DATE"] else 1.0,
        'school_match': 1.5 if query_entities["ORG"] else 1.0,
        'type_match': 1.3,
        'title_technique_match': 1.2
    }
    
    # Dictionary to store all items with their scores
    all_items = {}
    
    # Process SQLite results
    for rank, result in enumerate(sqlite_results):
        item_id = f"artwork:{result['id']}"
        boost = match_type_boosts.get(result.get('match_type', ''), 1.0)
        
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'SQLite',
                'id': result['id'],
                'image_file': result['image_file'],
                'title': result['title'],
                'description': result.get('description', ''),
                'timeframe': result.get('timeframe', ''),
                'author_name': result.get('author_name', ''),
                'type_name': result.get('type_name', ''),
                'school_name': result.get('school_name', ''),
                'match_type': result.get('match_type', ''),
                'rrf_score': 0
            }
        
        # Add RRF score with weight
        all_items[item_id]['rrf_score'] += weights['sqlite'] * boost * (1 / (k + rank))
    
    # Process sentence results
    for rank, result in enumerate(sentence_results):
        item_id = f"artwork:{result['artwork_id']}"
        
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'Sentence Match',
                'id': result['artwork_id'],
                'image_file': result['image_file'],
                'title': result['title'],
                'matched_sentence': result['sentence'],
                'rrf_score': 0
            }
        else:
            # Add the matched sentence if not already there
            if 'matched_sentence' not in all_items[item_id]:
                all_items[item_id]['matched_sentence'] = result['sentence']
        
        # Add RRF score with weight
        all_items[item_id]['rrf_score'] += weights['sentences'] * (1 / (k + rank))
    
    # Process HNSW results
    for rank, result in enumerate(hnsw_results):
        item_id = f"artwork:{result['id']}"
        
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'HNSW Similarity',
                'id': result['id'],
                'image_file': result['image_file'],
                'title': result['title'],
                'description': result.get('description', ''),
                'author_name': result.get('author_name', ''),
                'type_name': result.get('type_name', ''),
                'timeframe': result.get('timeframe', ''),
                'rrf_score': 0
            }
        
        # Add RRF score with weight
        all_items[item_id]['rrf_score'] += weights['hnsw'] * (1 / (k + rank))
    
    # Process KG results
    for rank, result in enumerate(kg_results):
        item_id = result['subject']  # Use subject URI as ID
        
        if item_id not in all_items:
            all_items[item_id] = {
                'source': 'Knowledge Graph',
                'subject': result['subject'],
                'predicate': result['predicate'],
                'object': result['object'],
                'subject_label': result.get('subject_label', ''),
                'predicate_label': result.get('predicate_label', ''),
                'object_label': result.get('object_label', ''),
                'rrf_score': 0
            }
        
        # Add RRF score with weight
        all_items[item_id]['rrf_score'] += weights['kg'] * (1 / (k + rank))
    
    # Convert dictionary to list and sort by RRF score
    ranked_results = list(all_items.values())
    ranked_results.sort(key=lambda x: x['rrf_score'], reverse=True)
    
    return ranked_results[:20]  # Return top 20 results

    
# First fix sanitize_fts_query to preserve entity names
def sanitize_fts_query(query):
    """Sanitize a query string for FTS5 while preserving named entities."""
    # Extract entities before sanitization
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    
    # Remove question marks, asterisks, quotes and other special characters
    sanitized = re.sub(r'[?*"\'\(\):]', ' ', query)
    # Remove extra spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Only remove stop words from parts that aren't entity names
    stop_words = set(nlp.Defaults.stop_words)
    if entities:
        # If we found entities, keep them intact
        words = []
        for word in sanitized.split():
            # Keep the word if it's part of an entity or not a stop word
            if any(word.lower() in entity.lower() for entity in entities) or word.lower() not in stop_words:
                words.append(word)
        sanitized = ' '.join(words)
    else:
        # Otherwise just filter stop words
        sanitized = ' '.join(word for word in sanitized.split() if word.lower() not in stop_words)
    
    return sanitized

if __name__ == "__main__":
    print("Starting retrieval...")
    original_query = "BIMBI connection"
    query = sanitize_fts_query(original_query)
    print(f"Sanitized query: '{query}'")
    results = hybrid_retrieve(query)
    print(f"Retrieved {len(results)} results.")
    for result in results:
        print(result)
    print("Retrieval complete.")
    
    kg = Graph()
    kg.parse(KG_FILE, format="turtle")
    

    for s, p, o in kg:
        if "bimbi" in str(s) or "BIMBI" in str(o):
            print(f"KG triple: {s} {p} {o}")
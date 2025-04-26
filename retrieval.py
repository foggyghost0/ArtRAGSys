import sqlite3
import csv
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import re

class ArtSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the FAISS index and ensure it's properly configured
        try:
            print("Loading FAISS index...")
            self.index = faiss.read_index('art_vectors.faiss')
            
            # Check if index is empty
            ntotal = self.index.ntotal
            print(f"FAISS index contains {ntotal} vectors")
            
            # If index is empty or doesn't have proper ID mapping, rebuild it
            if ntotal == 0:
                print("FAISS index is empty - will need to be rebuilt")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            
        self.conn = sqlite3.connect('art_database.db')
    
    def semantic_search(self, query, k=5):
        """Find semantically similar art descriptions"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        # Debug information
        print(f"FAISS search returned indices: {indices[0]}")
        
        results = []
        for idx in indices[0]:
            print(f"Looking up ID: {idx} in database...")
            cursor = self.conn.execute('''
            SELECT s.sentence_text, a.image_file, a.title, a.author, a.type, a.school, s.id
            FROM description_sentences s
            JOIN artworks a ON s.artwork_id = a.image_file
            WHERE s.id = ?
            ''', (int(idx),))
            result = cursor.fetchone()
            if result:
                print(f"Found match for ID {idx}: {result[0][:50]}...")
                results.append({
                    'sentence': result[0],
                    'image_file': result[1],
                    'title': result[2],
                    'author': result[3],
                    'type': result[4],
                    'school': result[5],
                    'id': int(idx)
                })
            else:
                print(f"No database match found for ID {idx}")
                
                # Try to find any record in description_sentences with this ID
                check_cursor = self.conn.execute('''
                SELECT id FROM description_sentences WHERE id = ?
                ''', (int(idx),))
                check_result = check_cursor.fetchone()
                if check_result:
                    print(f"ID {idx} exists in description_sentences but join failed")
                else:
                    print(f"ID {idx} does not exist in description_sentences table")
        
        return results
    
    def hybrid_search(self, text_query, semantic_query, k=5):
        """Combine text and semantic search"""
        # Text search
        cursor = self.conn.execute('''
        SELECT s.id, s.sentence_text, a.image_file, a.title, a.author
        FROM sentences_fts f
        JOIN description_sentences s ON f.rowid = s.id
        JOIN artworks a ON s.artwork_id = a.image_file
        WHERE f.sentence_text MATCH ?
        ORDER BY rank
        LIMIT ?
        ''', (text_query, k*2))
        text_results = cursor.fetchall()
        
        # Semantic search
        semantic_results = self.semantic_search(semantic_query, k*2)
        
        # Combine and deduplicate
        combined = {(r['id'], r['sentence']) for r in semantic_results}
        combined.update({(r[0], r[1]) for r in text_results})
        
        return list(combined)[:k]


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
        return hybrid_results

if __name__ == '__main__':
    main_search("flower", "still life with flowers")
    
    #implement BM25 ranking? 
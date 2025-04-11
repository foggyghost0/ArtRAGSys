import sqlite3
import csv
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
nltk.download('punkt')

def safe_split(value, delimiter=',', maxsplit=1, default=('', '')):
    """Safely split strings with default fallback"""
    try:
        parts = value.split(delimiter, maxsplit)
        return tuple(p.strip() for p in parts) + (default[len(parts):])
    except:
        return default

def parse_timeframe(timeframe):
    """Parse timeframe into start/end years"""
    try:
        if not timeframe:
            return None, None
        if '-' in timeframe:
            start, end = map(int, timeframe.split('-'))
            return start, end
        elif re.match(r'^\d{4}$', timeframe):  # Single year
            year = int(timeframe)
            return year, year
        else:
            return None, None
    except:
        return None, None

def create_databases():
    # Initialize embedding model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dimension = model.get_sentence_embedding_dimension()
    
    # Initialize FAISS index
    print("Creating FAISS index...")
    # Create the base index
    base_index = faiss.IndexFlatIP(embedding_dimension)
    # Wrap it with IndexIDMap to support custom IDs
    faiss_index = faiss.IndexIDMap(base_index)
    id_list = []
    text_list = []
    
    # Create SQLite database
    print("Creating SQLite database...")
    conn = sqlite3.connect('art_database.db')
    conn.execute('PRAGMA foreign_keys = ON;')
    cursor = conn.cursor()

    # Create artworks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS artworks (
        image_file TEXT PRIMARY KEY NOT NULL,
        author TEXT,
        title TEXT NOT NULL,
        medium TEXT,
        dimensions TEXT,
        date TEXT,
        type TEXT,
        school TEXT,
        timeframe_start INTEGER,
        timeframe_end INTEGER
    )''')

    # Create description sentences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS description_sentences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artwork_id TEXT NOT NULL,
        sentence_text TEXT NOT NULL,
        sentence_order INTEGER NOT NULL,
        FOREIGN KEY (artwork_id) REFERENCES artworks(image_file) ON DELETE CASCADE
    )''')

    # Create FTS virtual table for text search
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts 
    USING fts5(sentence_text, content='description_sentences', content_rowid='id')
    ''')

    # Create triggers to maintain FTS index
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS description_ai AFTER INSERT ON description_sentences BEGIN
        INSERT INTO sentences_fts(rowid, sentence_text) VALUES (new.id, new.sentence_text);
    END''')

    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS description_ad AFTER DELETE ON description_sentences BEGIN
        INSERT INTO sentences_fts(sentences_fts, rowid, sentence_text) VALUES ('delete', old.id, old.sentence_text);
    END''')

    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS description_au AFTER UPDATE ON description_sentences BEGIN
        INSERT INTO sentences_fts(sentences_fts, rowid, sentence_text) VALUES ('delete', old.id, old.sentence_text);
        INSERT INTO sentences_fts(rowid, sentence_text) VALUES (new.id, new.sentence_text);
    END''')

    # Process CSV file
    print("Processing CSV data...")
    with open('main_data.csv', 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            try:
                # Validate required fields
                if not row.get('IMAGE_FILE') or not row.get('TITLE'):
                    print(f"Skipping row {row_num}: Missing required fields")
                    continue

                # Process author name
                author = None
                if row.get('AUTHOR'):
                    if ', ' in row['AUTHOR']:
                        last, first = row['AUTHOR'].split(', ', 1)
                        author = f"{first.strip()} {last.strip()}"
                    else:
                        author = row['AUTHOR']

                # Process technique
                technique = row.get('TECHNIQUE', '')
                medium, dimensions = safe_split(technique, ', ', 1)

                # Process timeframe
                start_year, end_year = parse_timeframe(row.get('TIMEFRAME', ''))

                # Insert artwork
                cursor.execute('''
                INSERT INTO artworks VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )''', (
                    row['IMAGE_FILE'],
                    author,
                    row['TITLE'],
                    medium or None,
                    dimensions or None,
                    row.get('DATE'),
                    row.get('TYPE'),
                    row.get('SCHOOL'),
                    start_year,
                    end_year
                ))

                # Process description sentences
                description = row.get('DESCRIPTION', '')
                sentences = sent_tokenize(description) if description else []
                for i, sentence in enumerate(sentences, 1):
                    cursor.execute('''
                    INSERT INTO description_sentences (artwork_id, sentence_text, sentence_order)
                    VALUES (?, ?, ?)
                    ''', (row['IMAGE_FILE'], sentence.strip(), i))
                    
                    # Store for vectorization
                    sentence_id = cursor.lastrowid
                    id_list.append(sentence_id)
                    text_list.append(sentence.strip())

            except Exception as e:
                print(f"Error processing row {row_num}: {str(e)}")
                conn.rollback()
                continue

    # Create indexes
    print("Creating database indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_author ON artworks(author)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON artworks(timeframe_start, timeframe_end)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_school ON artworks(school)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_artwork_id ON description_sentences(artwork_id)')

    conn.commit()

    # Generate embeddings in batches
    print(f"Generating embeddings for {len(text_list)} sentences...")
    batch_size = 64
    embeddings = np.zeros((len(text_list), embedding_dimension), dtype=np.float32)
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        batch_emb = model.encode(batch_texts, show_progress_bar=False)
        embeddings[i:i+batch_size] = batch_emb
        print(f"Processed {min(i+batch_size, len(text_list))}/{len(text_list)} sentences")

    # Add to FAISS index with SQLite IDs
    print("Building FAISS index...")
    faiss_index.add_with_ids(embeddings, np.array(id_list, dtype=np.int64))
    
    # Save FAISS index
    print("Saving FAISS index...")
    faiss.write_index(faiss_index, 'art_vectors.faiss')
    
    # Save ID mapping for reference
    with open('id_mapping.txt', 'w') as f:
        for idx, text in zip(id_list, text_list):
            f.write(f"{idx}\t{text[:100]}\n")

    print("Database and vector index creation complete!")
    conn.close()

class ArtSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index('art_vectors.faiss')
        self.conn = sqlite3.connect('art_database.db')
    
    def semantic_search(self, query, k=5):
        """Find semantically similar art descriptions"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            cursor = self.conn.execute('''
            SELECT s.sentence_text, a.image_file, a.title, a.author, a.type, a.school
            FROM description_sentences s
            JOIN artworks a ON s.artwork_id = a.image_file
            WHERE s.id = ?
            ''', (int(idx),))
            result = cursor.fetchone()
            if result:
                results.append({
                    'sentence': result[0],
                    'image_file': result[1],
                    'title': result[2],
                    'author': result[3],
                    'type': result[4],
                    'school': result[5],
                    'id': int(idx)
                })
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

if __name__ == '__main__':
    create_databases()
    # #Example usage:
    # search = ArtSearch()
    # results = search.semantic_search("royal portrait painting")
    # for r in results:
    #     print(f"{r['title']} by {r['author']}: {r['sentence']}")
        
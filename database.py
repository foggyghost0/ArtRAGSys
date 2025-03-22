import sqlite3
import csv
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download all required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ensure all necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Create database
conn = sqlite3.connect('artrag.db')
cursor = conn.cursor()

# Create tables (using schema defined above)
cursor.execute('''
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS artwork_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS schools (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS artworks (
    id INTEGER PRIMARY KEY,
    image_file TEXT NOT NULL,
    title TEXT NOT NULL,
    author_id INTEGER,
    description TEXT,
    technique TEXT,
    date TEXT,
    type_id INTEGER,
    school_id INTEGER,
    timeframe TEXT,
    FOREIGN KEY (author_id) REFERENCES authors(id),
    FOREIGN KEY (type_id) REFERENCES artwork_types(id),
    FOREIGN KEY (school_id) REFERENCES schools(id)
)''')

# New table for sentence-segmented descriptions
cursor.execute('''
CREATE TABLE IF NOT EXISTS description_sentences (
    id INTEGER PRIMARY KEY,
    artwork_id INTEGER NOT NULL,
    sentence TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,
    FOREIGN KEY (artwork_id) REFERENCES artworks(id)
)''')

# Index for efficient sentence retrieval
cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_artwork ON description_sentences(artwork_id)')

# Create indexes for efficient querying
cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_author ON artworks(author_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_type ON artworks(type_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_school ON artworks(school_id)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_timeframe ON artworks(timeframe)')

# Enable FTS5 for full-text search (if supported)
try:
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS artwork_search USING fts5(
        title, description, technique,
        content='artworks',
        content_rowid='id'
    )''')
    
    # Also add FTS for sentences
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS sentence_search USING fts5(
        sentence,
        content='description_sentences',
        content_rowid='id'
    )''')
    
    # Triggers to keep search index updated
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS artworks_ai AFTER INSERT ON artworks BEGIN
        INSERT INTO artwork_search(rowid, title, description, technique)
        VALUES (new.id, new.title, new.description, new.technique);
    END''')
    
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS artworks_au AFTER UPDATE ON artworks BEGIN
        INSERT INTO artwork_search(artwork_search, rowid, title, description, technique)
        VALUES('delete', old.id, old.title, old.description, old.technique);
        INSERT INTO artwork_search(rowid, title, description, technique)
        VALUES (new.id, new.title, new.description, new.technique);
    END''')
    
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS artworks_ad AFTER DELETE ON artworks BEGIN
        INSERT INTO artwork_search(artwork_search, rowid, title, description, technique)
        VALUES('delete', old.id, old.title, old.description, old.technique);
    END''')
    
    # Triggers for sentence search
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS sentences_ai AFTER INSERT ON description_sentences BEGIN
        INSERT INTO sentence_search(rowid, sentence)
        VALUES (new.id, new.sentence);
    END''')
    
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS sentences_au AFTER UPDATE ON description_sentences BEGIN
        INSERT INTO sentence_search(sentence_search, rowid, sentence)
        VALUES('delete', old.id, old.sentence);
        INSERT INTO sentence_search(rowid, sentence)
        VALUES (new.id, new.sentence);
    END''')
    
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS sentences_ad AFTER DELETE ON description_sentences BEGIN
        INSERT INTO sentence_search(sentence_search, rowid, sentence)
        VALUES('delete', old.id, old.sentence);
    END''')
    
    print("Full-text search enabled for artworks and sentences")
except sqlite3.OperationalError as e:
    print(f"Warning: FTS5 not fully supported in this SQLite version, full-text search limited: {e}")

# Function to segment a description into sentences
def segment_description(description):
    if not description:
        return []
    
    # Clean up the text a bit (remove extra spaces, fix common issues)
    description = re.sub(r'\s+', ' ', description).strip()
    
    # Use NLTK's sentence tokenizer
    sentences = sent_tokenize(description)
    
    # Post-process to clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# Read the CSV file
with open('main_data.csv', 'r', encoding='latin1') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Process each artwork
    for row in reader:
        # Insert or get author ID
        cursor.execute('INSERT OR IGNORE INTO authors (name) VALUES (?)', (row['AUTHOR'],))
        cursor.execute('SELECT id FROM authors WHERE name = ?', (row['AUTHOR'],))
        author_id = cursor.fetchone()[0]
        
        # Insert or get type ID
        cursor.execute('INSERT OR IGNORE INTO artwork_types (name) VALUES (?)', (row['TYPE'],))
        cursor.execute('SELECT id FROM artwork_types WHERE name = ?', (row['TYPE'],))
        type_id = cursor.fetchone()[0]
        
        # Insert or get school ID
        cursor.execute('INSERT OR IGNORE INTO schools (name) VALUES (?)', (row['SCHOOL'],))
        cursor.execute('SELECT id FROM schools WHERE name = ?', (row['SCHOOL'],))
        school_id = cursor.fetchone()[0]
        
        # Insert artwork
        cursor.execute('''
            INSERT INTO artworks (image_file, title, author_id, description, 
                                 technique, date, type_id, school_id, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['IMAGE_FILE'], row['TITLE'], author_id, row['DESCRIPTION'],
            row['TECHNIQUE'], row['DATE'], type_id, school_id, row['TIMEFRAME']
        ))
        
        # Get the ID of the inserted artwork
        artwork_id = cursor.lastrowid
        
        # Segment the description into sentences and insert them
        sentences = segment_description(row['DESCRIPTION'])
        for idx, sentence in enumerate(sentences):
            cursor.execute('''
                INSERT INTO description_sentences (artwork_id, sentence, sentence_index)
                VALUES (?, ?, ?)
            ''', (artwork_id, sentence, idx))
    
    conn.commit()
conn.close()

print("Database creation complete with sentence segmentation")
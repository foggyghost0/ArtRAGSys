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

# Database file path
DB_FILE = 'artrag.db'

# CSV file path
CSV_FILE = 'main_data.csv'

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

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """Create a table from the create_table_sql statement."""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def main():
    database = DB_FILE

    sql_create_authors_table = """
    CREATE TABLE IF NOT EXISTS authors (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    """

    sql_create_artwork_types_table = """
    CREATE TABLE IF NOT EXISTS artwork_types (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    """

    sql_create_schools_table = """
    CREATE TABLE IF NOT EXISTS schools (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    """

    sql_create_artworks_table = """
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
    );
    """
    
    sql_create_description_sentences_table = """
    CREATE TABLE IF NOT EXISTS description_sentences (
        id INTEGER PRIMARY KEY,
        artwork_id INTEGER NOT NULL,
        sentence TEXT NOT NULL,
        sentence_index INTEGER NOT NULL,
        FOREIGN KEY (artwork_id) REFERENCES artworks(id)
    );
    """

    # Create a database connection
    conn = create_connection(database)

    # Create tables
    if conn is not None:
        create_table(conn, sql_create_authors_table)
        create_table(conn, sql_create_artwork_types_table)
        create_table(conn, sql_create_schools_table)
        create_table(conn, sql_create_artworks_table)
        create_table(conn, sql_create_description_sentences_table)
        
        cursor = conn.cursor()
        
        # Index for efficient sentence retrieval
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_artwork ON description_sentences(artwork_id)')

        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_author ON artworks(author_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_type ON artworks(type_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_school ON artworks(school_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_timeframe ON artworks(timeframe)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artworks_date ON artworks(date)')

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

        # Read the CSV file
        with open(CSV_FILE, 'r', encoding='latin-1') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Prepare data for batch insertion
            authors_data = []
            artwork_types_data = []
            schools_data = []
            artworks_data = []
            description_sentences_data = []
            
            # Process each artwork
            for row in reader:
                try:
                    # Data validation
                    author_name = row['AUTHOR']
                    artwork_type = row['TYPE']
                    school_name = row['SCHOOL']
                    
                    if not all([author_name, artwork_type, school_name]):
                        print(f"Skipping row due to missing data: {row}")
                        continue
                    
                    # Insert or get author ID
                    authors_data.append((author_name,))
                    
                    # Insert or get type ID
                    artwork_types_data.append((artwork_type,))
                    
                    # Insert or get school ID
                    schools_data.append((school_name,))
                    
                    # Artwork data
                    artworks_data.append((
                        row['IMAGE_FILE'], row['TITLE'], author_name, row['DESCRIPTION'],
                        row['TECHNIQUE'], row['DATE'], artwork_type, school_name, row['TIMEFRAME']
                    ))
                    
                    # Segment the description into sentences and prepare for insertion
                    sentences = segment_description(row['DESCRIPTION'])
                    for idx, sentence in enumerate(sentences):
                        description_sentences_data.append((row['IMAGE_FILE'], sentence, idx))
                
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")
                    continue
            
            try:
                # Insert authors
                cursor.executemany('INSERT OR IGNORE INTO authors (name) VALUES (?)', authors_data)
                
                # Insert artwork types
                cursor.executemany('INSERT OR IGNORE INTO artwork_types (name) VALUES (?)', artwork_types_data)
                
                # Insert schools
                cursor.executemany('INSERT OR IGNORE INTO schools (name) VALUES (?)', schools_data)
                
                # Fetch author, type, and school IDs
                author_ids = {name: id for id, name in cursor.execute('SELECT id, name FROM authors')}
                type_ids = {name: id for id, name in cursor.execute('SELECT id, name FROM artwork_types')}
                school_ids = {name: id for id, name in cursor.execute('SELECT id, name FROM schools')}
                
                # Insert artworks
                artworks_to_insert = []
                for row in artworks_data:
                    author_id = author_ids[row[2]]
                    type_id = type_ids[row[6]]
                    school_id = school_ids[row[7]]
                    artworks_to_insert.append((row[0], row[1], author_id, row[3], row[4], row[5], type_id, school_id, row[8]))
                
                cursor.executemany('''
                    INSERT INTO artworks (image_file, title, author_id, description, 
                                         technique, date, type_id, school_id, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', artworks_to_insert)
                
                # Get artwork IDs and insert sentences
                artwork_ids = {image_file: id for id, image_file in cursor.execute('SELECT id, image_file FROM artworks')}
                sentences_to_insert = []
                for image_file, sentence, idx in description_sentences_data:
                    artwork_id = artwork_ids[image_file]
                    sentences_to_insert.append((artwork_id, sentence, idx))
                
                cursor.executemany('''
                    INSERT INTO description_sentences (artwork_id, sentence, sentence_index)
                    VALUES (?, ?, ?)
                ''', sentences_to_insert)
                
                conn.commit()
                print("Database creation complete with sentence segmentation")
            
            except sqlite3.Error as e:
                print(f"Database insertion error: {e}")
            
            except Exception as e:
                print(f"General error during data processing: {e}")
            
            finally:
                conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    main()
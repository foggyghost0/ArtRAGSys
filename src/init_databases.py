"""
Database initialization module for Art RAG System.
Creates SQLite database, ChromaDB vector storage, and processes artwork data.
"""

import sqlite3
import csv
import re
from typing import Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

def safe_split(
    value: str,
    delimiter: str = ",",
    maxsplit: int = 1,
    default: Tuple[str, ...] = ("", ""),
) -> Tuple[str, ...]:
    """Safely split strings with default fallback."""
    try:
        if not value:
            return default
        parts = value.split(delimiter, maxsplit)
        return tuple(p.strip() for p in parts) + default[len(parts) :]
    except (AttributeError, ValueError) as e:
        print(f"Error splitting value '{value}': {e}")
        return default


def parse_timeframe(timeframe: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse timeframe into start/end years."""
    try:
        if not timeframe:
            return None, None

        # Handle range format (e.g., "1601-1650")
        if "-" in timeframe:
            start_str, end_str = timeframe.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            return start, end

        # Handle single year format (e.g., "1650")
        elif re.match(r"^\d{4}$", timeframe.strip()):
            year = int(timeframe.strip())
            return year, year

        else:
            return None, None
    except (ValueError, AttributeError) as e:
        print(f"Error parsing timeframe '{timeframe}': {e}")
        return None, None


def parse_author_name(author: str) -> Optional[str]:
    """Parse author name from 'Last, First' to 'First Last' format."""
    try:
        if not author:
            return None

        if ", " in author:
            last, first = author.split(", ", 1)
            return f"{first.strip()} {last.strip()}"
        else:
            return author.strip()
    except (AttributeError, ValueError) as e:
        print(f"Error parsing author name '{author}': {e}")
        return author


def create_sqlite_tables(cursor: sqlite3.Cursor) -> None:
    """Create SQLite database tables with proper indexing."""
    print("Creating SQLite database tables...")

    # Create artworks table - using image_file as primary key (unique artwork ID)
    cursor.execute(
        """
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
        timeframe_end INTEGER,
        full_description TEXT NOT NULL
    )"""
    )

    # Create description sentences table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS description_sentences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artwork_id TEXT NOT NULL,
        sentence_text TEXT NOT NULL,
        sentence_order INTEGER NOT NULL,
        FOREIGN KEY (artwork_id) REFERENCES artworks(image_file) ON DELETE CASCADE
    )"""
    )

    # Create FTS virtual table for text search
    cursor.execute(
        """
    CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts 
    USING fts5(sentence_text, content='description_sentences', content_rowid='id')
    """
    )

    # Create triggers to maintain FTS index
    cursor.execute(
        """
    CREATE TRIGGER IF NOT EXISTS description_ai AFTER INSERT ON description_sentences BEGIN
        INSERT INTO sentences_fts(rowid, sentence_text) VALUES (new.id, new.sentence_text);
    END"""
    )

    cursor.execute(
        """
    CREATE TRIGGER IF NOT EXISTS description_ad AFTER DELETE ON description_sentences BEGIN
        INSERT INTO sentences_fts(sentences_fts, rowid, sentence_text) VALUES ('delete', old.id, old.sentence_text);
    END"""
    )

    cursor.execute(
        """
    CREATE TRIGGER IF NOT EXISTS description_au AFTER UPDATE ON description_sentences BEGIN
        INSERT INTO sentences_fts(sentences_fts, rowid, sentence_text) VALUES ('delete', old.id, old.sentence_text);
        INSERT INTO sentences_fts(rowid, sentence_text) VALUES (new.id, new.sentence_text);
    END"""
    )

    # Create database indexes for efficient querying
    print("Creating database indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_author ON artworks(author)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timeframe ON artworks(timeframe_start, timeframe_end)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_school ON artworks(school)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON artworks(type)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_artwork_id ON description_sentences(artwork_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentence_order ON description_sentences(sentence_order)"
    )


def initialize_chromadb() -> chromadb.Collection:
    """Initialize ChromaDB vector database."""
    print("Initializing ChromaDB...")

    # Create ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path="./chroma_db")

    # Get or create collections for different types of embeddings
    # Collection for individual sentences
    try:
        sentence_collection = client.get_collection(name="artwork_sentences")
        print("Found existing sentence collection, deleting to recreate...")
        client.delete_collection(name="artwork_sentences")
    except Exception:
        pass

    sentence_collection = client.create_collection(
        name="artwork_sentences", metadata={"hnsw:space": "cosine"}
    )

    # Collection for full descriptions
    try:
        description_collection = client.get_collection(name="artwork_descriptions")
        print("Found existing description collection, deleting to recreate...")
        client.delete_collection(name="artwork_descriptions")
    except Exception:
        pass

    description_collection = client.create_collection(
        name="artwork_descriptions", metadata={"hnsw:space": "cosine"}
    )

    return sentence_collection, description_collection


def process_csv_data(
    csv_path: str,
    cursor: sqlite3.Cursor,
    model: SentenceTransformer,
    sentence_collection: chromadb.Collection,
    description_collection: chromadb.Collection,
) -> None:
    """Process CSV data and populate databases."""
    print("Processing CSV data...")

    sentence_texts = []
    sentence_ids = []
    sentence_metadata = []

    description_texts = []
    description_ids = []
    description_metadata = []

    processed_count = 0
    error_count = 0

    with open(csv_path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, 1):
            try:
                # Validate required fields
                if not row.get("IMAGE_FILE") or not row.get("TITLE"):
                    print(f"Skipping row {row_num}: Missing required fields")
                    error_count += 1
                    continue

                # Process fields
                author = parse_author_name(row.get("AUTHOR", ""))
                technique = row.get("TECHNIQUE", "")
                medium, dimensions = safe_split(technique, ", ", 1)
                start_year, end_year = parse_timeframe(row.get("TIMEFRAME", ""))
                description = row.get("DESCRIPTION", "").strip()

                # Insert artwork into SQLite
                cursor.execute(
                    """
                INSERT OR REPLACE INTO artworks VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                    (
                        row["IMAGE_FILE"],
                        author,
                        row["TITLE"],
                        medium or None,
                        dimensions or None,
                        row.get("DATE"),
                        row.get("TYPE"),
                        row.get("SCHOOL"),
                        start_year,
                        end_year,
                        description,
                    ),
                )

                # Process description for full-text embedding
                if description:
                    description_texts.append(description)
                    description_ids.append(f"desc_{row['IMAGE_FILE']}")
                    description_metadata.append(
                        {
                            "artwork_id": row["IMAGE_FILE"],
                            "title": row["TITLE"],
                            "author": author or "Unknown",
                            "type": row.get("TYPE", ""),
                            "school": row.get("SCHOOL", ""),
                            "content_type": "full_description",
                        }
                    )

                # Process description sentences
                sentences = sent_tokenize(description) if description else []
                for i, sentence in enumerate(sentences, 1):
                    sentence = sentence.strip()
                    if sentence:
                        cursor.execute(
                            """
                        INSERT INTO description_sentences (artwork_id, sentence_text, sentence_order)
                        VALUES (?, ?, ?)
                        """,
                            (row["IMAGE_FILE"], sentence, i),
                        )

                        # Prepare for batch embedding
                        sentence_id = cursor.lastrowid
                        sentence_texts.append(sentence)
                        sentence_ids.append(f"sent_{sentence_id}")
                        sentence_metadata.append(
                            {
                                "sentence_id": sentence_id,
                                "artwork_id": row["IMAGE_FILE"],
                                "title": row["TITLE"],
                                "author": author or "Unknown",
                                "type": row.get("TYPE", ""),
                                "school": row.get("SCHOOL", ""),
                                "sentence_order": i,
                                "content_type": "sentence",
                            }
                        )

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} artworks...")

            except Exception as e:
                print(f"Error processing row {row_num}: {e}")
                error_count += 1
                continue

    print(
        f"CSV processing complete. Processed: {processed_count}, Errors: {error_count}"
    )

    # Generate embeddings and add to ChromaDB in batches
    batch_size = 32

    # Process sentences
    if sentence_texts:
        print(f"Generating embeddings for {len(sentence_texts)} sentences...")
        for i in range(0, len(sentence_texts), batch_size):
            batch_texts = sentence_texts[i : i + batch_size]
            batch_ids = sentence_ids[i : i + batch_size]
            batch_metadata = sentence_metadata[i : i + batch_size]

            embeddings = model.encode(batch_texts, show_progress_bar=False)

            sentence_collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids,
            )

            print(
                f"Added sentence batch {i//batch_size + 1}/{(len(sentence_texts) + batch_size - 1)//batch_size}"
            )

    # Process full descriptions
    if description_texts:
        print(
            f"Generating embeddings for {len(description_texts)} full descriptions..."
        )
        for i in range(0, len(description_texts), batch_size):
            batch_texts = description_texts[i : i + batch_size]
            batch_ids = description_ids[i : i + batch_size]
            batch_metadata = description_metadata[i : i + batch_size]

            embeddings = model.encode(batch_texts, show_progress_bar=False)

            description_collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids,
            )

            print(
                f"Added description batch {i//batch_size + 1}/{(len(description_texts) + batch_size - 1)//batch_size}"
            )


def create_databases(
    csv_path: str = "../data/main_data.csv", db_path: str = "art_database.db"
) -> None:
    """Main function to create and populate all databases."""
    print("Starting database initialization...")

    # Initialize embedding model
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize ChromaDB
    sentence_collection, description_collection = initialize_chromadb()

    # Create SQLite database
    print("Creating SQLite database...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    try:
        # Create tables and indexes
        create_sqlite_tables(cursor)

        # Process CSV data and populate both databases
        process_csv_data(
            csv_path, cursor, model, sentence_collection, description_collection
        )

        # Commit SQLite changes
        conn.commit()
        print("Database initialization completed successfully!")

        # Print summary statistics
        cursor.execute("SELECT COUNT(*) FROM artworks")
        artwork_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM description_sentences")
        sentence_count = cursor.fetchone()[0]

        print("\nSummary:")
        print(f"- Artworks: {artwork_count}")
        print(f"- Sentences: {sentence_count}")
        print(f"- ChromaDB sentence collection: {sentence_collection.count()}")
        print(f"- ChromaDB description collection: {description_collection.count()}")

    except Exception as e:
        print(f"Error during database creation: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Download required NLTK data
    print("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)

    # Create databases
    create_databases()

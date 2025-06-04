import sqlite3


def get_unique_schools_and_types(db_path="src/art_database.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT school FROM artworks WHERE school IS NOT NULL AND school != ''"
        )
        unique_schools = sorted([row[0] for row in cursor.fetchall()])

        cursor.execute(
            "SELECT DISTINCT type FROM artworks WHERE type IS NOT NULL AND type != ''"
        )
        unique_types = sorted([row[0] for row in cursor.fetchall()])

        conn.close()
        print(f"Found {len(unique_schools)} unique schools and {len(unique_types)} unique types.")
        print(f"Schools: {unique_schools}")
        print(f"Types: {unique_types}")
        return unique_schools, unique_types
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return [], []


if __name__ == "__main__":
    get_unique_schools_and_types()

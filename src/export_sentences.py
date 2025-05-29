import sqlite3
import csv
import os

def export_sentences_to_csv(output_filename='sentences_for_evaluation.csv'):
    """
    Exports all sentences from the art database to a CSV file with their IDs
    and related artwork information to help create evaluation queries.
    
    Args:
        output_filename (str): Name of the output CSV file
    
    Returns:
        int: Number of sentences exported
    """
    print(f"Connecting to database and exporting sentences to {output_filename}...")
    
    # Connect to the SQLite database
    conn = sqlite3.connect('art_database.db')
    cursor = conn.cursor()
    
    # Query to get all sentences with artwork details
    cursor.execute('''
    SELECT 
        s.id,
        s.sentence_text,
        a.image_file,
        a.title,
        a.author,
        a.type,
        a.school,
        a.timeframe_start,
        a.timeframe_end
    FROM 
        description_sentences s
    JOIN 
        artworks a ON s.artwork_id = a.image_file
    ORDER BY 
        s.id
    ''')
    
    results = cursor.fetchall()
    
    # Write to CSV
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            'sentence_id', 
            'sentence_text', 
            'image_file', 
            'title', 
            'author', 
            'type', 
            'school', 
            'timeframe_start', 
            'timeframe_end'
        ])
        
        # Write data rows
        for row in results:
            writer.writerow(row)
    
    print(f"Successfully exported {len(results)} sentences to {output_filename}")
    
    # Sample the first few entries to help with creating queries
    print("\nSample entries (first 5):")
    for i in range(min(5, len(results))):
        print(f"ID: {results[i][0]}, Text: {results[i][1][:100]}...")
        print(f"  Artwork: {results[i][3]} by {results[i][4] or 'Unknown'}")
        print("---")
    
    conn.close()
    return len(results)

def create_evaluation_template(output_filename='evaluation_queries_template.csv'):
    """
    Creates a template CSV file for evaluation queries
    """
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Query', 'Expected Document IDs (comma-separated)'])
        writer.writerow(['Renaissance portraits', ''])
        writer.writerow(['Religious symbolism in paintings', ''])
        writer.writerow(['Landscape paintings with mountains', ''])
        writer.writerow(['Still life with flowers', ''])
        writer.writerow(['Mythological scenes', ''])
    
    print(f"Created evaluation template at {output_filename}")
    print("Fill in the expected document IDs after reviewing the exported sentences")

if __name__ == '__main__':
    # Export all sentences
    count = export_sentences_to_csv()
    
    # Create an evaluation template if enough sentences were exported
    if count > 0:
        create_evaluation_template()
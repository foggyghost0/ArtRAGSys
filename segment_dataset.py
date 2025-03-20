import os
import csv
import nltk
from nltk.tokenize import sent_tokenize

# Ensure the necessary NLTK tokenizer models are downloaded
nltk.download('punkt')

# Fix to ensure we're using the standard punkt tokenizer
def segment_descriptions(input_file, output_file, encoding='latin-1'):
    """
    Segment descriptions in a CSV file into sentences.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        encoding: Encoding to use for the input file (try 'latin-1', 'cp1252', or 'iso-8859-1')
    """
    # First ensure correct punkt tokenizer is accessible
    try:
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
    except:
        # Fallback to simple sentence splitting
        def simple_sent_tokenize(text):
            return [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
    with open(input_file, mode='r', newline='', encoding=encoding) as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Input CSV file must have headers.")
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if 'DESCRIPTION' not in row or not row['DESCRIPTION']:
                    writer.writerow(row)  # Keep rows without descriptions
                    continue
                    
                description = row['DESCRIPTION']
                
                try:
                    if 'tokenizer' in locals():
                        sentences = tokenizer.tokenize(description)
                    else:
                        sentences = simple_sent_tokenize(description)
                except:
                    # Last resort fallback
                    sentences = [description]
                
                for sentence in sentences:
                    new_row = row.copy()
                    new_row['DESCRIPTION'] = sentence.strip()
                    writer.writerow(new_row)

# Example usage
input_csv = 'main_data.csv'
output_csv = 'segmented_data.csv'
segment_descriptions(input_csv, output_csv, encoding='latin-1')
import os 
import pandas as pd

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# delete unnecessary csv files
def delete_csv_files():
    data_path = os.path.join(script_dir, 'SemArt')
    data_files = os.listdir(data_path)
    data_files = [file for file in data_files if file.endswith('.csv') and "human" in file]
    data_files = [file for file in data_files if file.endswith('.txt')]
    for file in data_files:
        os.remove(os.path.join(data_path, file))

# fuse dataset
def fuse_datasets():
    # File paths
    val_file = os.path.join(script_dir, 'SemArt/semart_val.csv')
    train_file = os.path.join(script_dir, 'SemArt/semart_train.csv')
    test_file = os.path.join(script_dir, 'SemArt/semart_test.csv')
    
    # Expected column names
    expected_columns = ['IMAGE_FILE', 'DESCRIPTION', 'AUTHOR', 'TITLE', 'TECHNIQUE', 'DATE', 'TYPE', 'SCHOOL', 'TIMEFRAME']
    
    # Read the files with appropriate delimiters
    print("Reading files...")
    
    # Read val file (tab-separated)
    val_df = pd.read_csv(val_file, sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
    
    # Read train file (comma-separated)
    train_df = pd.read_csv(train_file, sep=',', encoding='ISO-8859-1', on_bad_lines='skip')
    
    # Read test file (tab-separated)
    test_df = pd.read_csv(test_file, sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
    
    # Create new dataframes with only the expected columns
    # This ensures all dataframes have exactly the same structure
    new_val_df = pd.DataFrame(columns=expected_columns)
    new_train_df = pd.DataFrame(columns=expected_columns)
    new_test_df = pd.DataFrame(columns=expected_columns)
    
    # Copy data from original dataframes to new ones
    for col in expected_columns:
        if col in val_df.columns:
            new_val_df[col] = val_df[col]
        if col in train_df.columns:
            new_train_df[col] = train_df[col]
        if col in test_df.columns:
            new_test_df[col] = test_df[col]
    
    # Concatenate the dataframes
    print("Concatenating dataframes...")
    combined_df = pd.concat([new_val_df, new_train_df, new_test_df], ignore_index=True)
    
    # Drop rows with any missing values
    print(f"Initial row count: {len(combined_df)}")
    combined_df.dropna(inplace=True)
    print(f"Row count after dropping NA: {len(combined_df)}")
    
    # Sort by IMAGE_FILE column
    combined_df.sort_values(by='IMAGE_FILE', inplace=True)
    
    # Verify we have only the expected columns
    print(f"Final columns: {combined_df.columns.tolist()}")
    
    return combined_df

delete_csv_files()
combined_dataset = fuse_datasets()

# Save the concatenated dataset to main_data.csv
output_file = os.path.join(script_dir, 'main_data.csv')
combined_dataset.to_csv(output_file, index=False, encoding='ISO-8859-1')

# Print the final number of correct rows
print(f"Number of correct rows: {len(combined_dataset)}")
print(combined_dataset.head())
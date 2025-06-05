import pandas as pd
df = pd.read_csv("data/main_data.csv", encoding="latin1")
print("Total rows:", len(df))
print("With description:", df['DESCRIPTION'].notnull().sum())
print("With image and title:", df[df['IMAGE_FILE'].notnull() & df['TITLE'].notnull()].shape[0])
print("Unique image files:", df['IMAGE_FILE'].nunique())

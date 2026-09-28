import pandas as pd
import numpy as np

# Load the dataset
file_path = '/Users/rajnishsingh/Desktop/comparision/dataset/jdt.csv'
df = pd.read_csv(file_path, sep=';')

# Clean column names (remove leading/trailing whitespace)
df.columns = [c.strip() for c in df.columns]

print("Columns:", df.columns.tolist())
print("\nBugs distribution:")
print(df['bugs'].value_counts())

print("\nNonTrivialBugs distribution:")
print(df['nonTrivialBugs'].value_counts())

# Check total samples
print(f"\nTotal samples: {len(df)}")

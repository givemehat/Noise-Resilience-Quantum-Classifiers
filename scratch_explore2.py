import pandas as pd
import os

dataset_dir = '/Users/rajnishsingh/Desktop/comparision/dataset'
for f in os.listdir(dataset_dir):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(dataset_dir, f), sep=';')
        df.columns = [c.strip() for c in df.columns]
        print(f"Dataset: {f}")
        print(f"Columns: {df.columns.tolist()}\n")

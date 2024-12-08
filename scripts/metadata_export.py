# File: metadata_export.py
import pandas as pd
import json

def export_metadata(dataset_path, metadata_path):
    # Load dataset
    dataset = pd.read_csv(dataset_path)

    # Extract column information
    metadata = {
        "columns": list(dataset.drop("target", axis=1).columns),  # Exclude the target column
        "target_column": "target"  # Replace with actual target column name
    }

    # Save metadata to JSON
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    dataset_path = "../datasets/cancer_dataset.csv"
    metadata_path = "../models/metadata.json"

    export_metadata(dataset_path, metadata_path)

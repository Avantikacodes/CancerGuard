# -*- coding: utf-8 -*-
"""model_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12BSNoaqPfvnMuIjtAT-Be8kppsMXEKaB
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import os  # To handle directory creation

def load_and_preprocess_data(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    print("Initial Dataset Preview:")
    print(df.head())

    # Check dataset shape
    print("Dataset shape:", df.shape)

    # Check for missing values
    print("\nChecking for Missing Values:")
    print(df.isnull().sum())

    # Drop rows where 'diagnosis' column is missing
    df = df.dropna(subset=['diagnosis'])
    print("After dropping rows with missing 'diagnosis' values, dataset shape:", df.shape)

    # Drop unnecessary columns (id and 'Unnamed: 32')
    df = df.drop(["Unnamed: 32", "id"], axis=1, errors="ignore")
    print("After dropping unnecessary columns, dataset shape:", df.shape)

    # Handle missing values in features (using mean imputation for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("After imputing missing values in feature columns, dataset shape:", df.shape)

    # Encode the target variable ('diagnosis' column: 'M' -> 1, 'B' -> 0)
    if 'diagnosis' in df.columns:
        print("Unique values before mapping:", df['diagnosis'].unique())
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    print("Unique values after mapping:", df['diagnosis'].unique())

    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Check if data is empty after preprocessing
    if X.empty or y.empty:
        print("Error: Empty dataset or target variable after preprocessing.")
        return None, None

    return X, y

def train_naive_bayes_model(X, y, model_path, scaler_path, metadata_path):
    # Check if data is empty
    if X.empty or y.empty:
        print("Error: Empty dataset or target variable.")
        return

    # Ensure the models directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directory '{model_dir}' created.")

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Save the model
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

    # Save metadata
    metadata = {"columns": list(X.columns), "target_column": "diagnosis"}
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
    print(f"Metadata saved to {metadata_path}")

    return model, X_test, y_test

if __name__ == "__main__":
    dataset_path = 'data.csv'  # Adjust to the correct path if needed
    model_path = './models/model.pkl'  # Ensure this path exists
    metadata_path = 'metadata.json'

    # Load and preprocess data
    X, y = load_and_preprocess_data(dataset_path)

    if X is not None and y is not None:
        # Train the model and save it
        model, X_test, y_test = train_naive_bayes_model(X, y, model_path, None, metadata_path)

        # Evaluating the model
        if model is not None:
            y_pred = model.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

            # Load metadata for feature names
            with open(metadata_path, "r") as file:
                metadata = json.load(file)
            feature_columns = metadata["columns"]

            # Create a test DataFrame with correct structure
            test_features = pd.DataFrame([[5.1, 3.5, 1.4, 0.2] + [0] * (len(feature_columns) - 4)],
                                         columns=feature_columns)

            # Predict
            predicted = model.predict(test_features)
            print("Predicted Class:", predicted)

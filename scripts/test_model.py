import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(model_path):
    # Load the trained model from the specified file
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: The file {model_path} does not exist.")
        return None

def load_metadata(metadata_path):
    # Load the metadata (columns and target variable)
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        print("Metadata loaded successfully.")
        return metadata
    except FileNotFoundError:
        print(f"Error: The file {metadata_path} does not exist.")
        return None

def load_and_preprocess_data(dataset_path, metadata, is_test=False):
    # Load the dataset
    df = pd.read_csv('data.csv')
    print("Dataset loaded successfully.")

    # Ensure that only the required columns are used
    required_columns = metadata['columns']
    df = df[required_columns]

    if not is_test:
        # For training data, we expect the target column 'diagnosis' to be present
        if 'diagnosis' not in df.columns:
            print("Error: Target column 'diagnosis' is missing from the training dataset.")
            return None, None
        y = df['diagnosis']  # Extract the target column
        X = df.drop('diagnosis', axis=1)  # Drop 'diagnosis' from the features
    else:
        # For test data, the target column 'diagnosis' may not be present
        X = df  # All the columns are features in the test dataset
        y = None  # No target column in the test dataset

    print("Features shape:", X.shape)
    return X, y

def test_model(model, X_test, y_test):
    # Make predictions using the model
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
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

if __name__ == "__main__":
    model_path = './models/model.pkl'  # Path to the saved model
    metadata_path = 'metadata.json'   # Path to the metadata file
    dataset_path = 'test_data.csv'    # Path to the test dataset

    # Load the trained model
    model = load_model(model_path)
    
    if model:
        # Load metadata to get the required feature columns
        metadata = load_metadata(metadata_path)

        if metadata:
            # Load and preprocess the test dataset
            X_test, y_test = load_and_preprocess_data(dataset_path, metadata, is_test=True)

            if X_test is not None:
                # Test the model
                if y_test is None:  # If there is no target column in the test data, only predict
                    print("Predicting without true target values (for evaluation only).")
                    predictions = model.predict(X_test)
                    print("Predictions:", predictions)
                else:
                    test_model(model, X_test, y_test)
            else:
                print("Error: Could not process the test dataset.")
        else:
            print("Error: Metadata not loaded.")
    else:
        print("Error: Model not loaded.")

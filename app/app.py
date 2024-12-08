import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import json

app = Flask(__name__)

# Path to the trained model and metadata
model_path = 'C:/Users/Avantika_07/Desktop/Project/models/model.pkl'  # Absolute path
metadata_path = 'C:/Users/Avantika_07/Desktop/Project/models/metadata.json'  # Absolute path

# Load the trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the metadata (column names)
with open(metadata_path, 'r') as file:
    metadata = json.load(file)

feature_columns = metadata["columns"]

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the form data (assuming you are sending only 4 feature values)
        input_data = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean'])
        ]
        
        # Ensure input_data has the same number of features as expected (30 in this case)
        # Append zeros for the missing features
        if len(input_data) < len(feature_columns):
            input_data.extend([0] * (len(feature_columns) - len(input_data)))

        # Convert input data into a DataFrame with the correct column names
        input_data = pd.DataFrame([input_data], columns=feature_columns)

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:  # The prediction output is an array, so we check the first element
            result = "Malignant"
        else:
            result = "Benign"

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

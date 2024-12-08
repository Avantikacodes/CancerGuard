from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model, scaler, and metadata
model_path = 'C:/Users/Avantika_07/Desktop/Project/models/model.pkl'  # Absolute path
metadata_path = 'C:/Users/Avantika_07/Desktop/Project/models/metadata.json'  # Absolute path

#model_path = './models/model.pkl'
#scaler_path = './models/scaler.pkl'
#metadata_path = './models/metadata.json'

# Load model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load scaler (if applicable)
#with open(scaler_path, 'rb') as file:
 #   scaler = pickle.load(file)

# Load metadata
import json
with open(metadata_path, 'r') as file:
    metadata = json.load(file)

# Route for the homepage where users input the data
@app.route('/')
def index():
    return render_template('index.html')

# Route to process the form data and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    form_data = request.form.to_dict()

    # Convert input data to a numpy array
    input_data = np.array([list(form_data.values())], dtype=float)

    # Ensure the correct order of features (as per metadata)
    feature_columns = metadata['columns']
    input_data = pd.DataFrame(input_data, columns=feature_columns)

    # Apply the same scaling to the input data as was done during model training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Map prediction (1: Malignant, 0: Benign)
    result = 'Malignant' if prediction[0] == 1 else 'Benign'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

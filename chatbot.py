import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure this path is correct)
model = load_model(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\models\gemstone_model.h5')

# Class labels for prediction
class_labels = ['Amethyst', 'Diamond', 'Emerald', 'Lapis Lazuli', 'Opal', 'Pearl', 'Quartz Smoky', 'Ruby', 'Sapphire Yellow', 'Topaz']

# Route to render the HTML page
@app.route("/")
def index():
    return render_template('index.html')  # Keep the same HTML rendering

# Route to handle image upload and prediction
@app.route("/predict", methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Preprocess the image for prediction
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match the model input

        # Predict using the model
        predictions = model.predict(img_array)
        percentages = predictions[0] * 100  # Get the percentages for each class

        # Convert the predictions to native Python float types for JSON serialization
        result = {class_labels[i]: round(float(percentages[i]), 2) for i in range(len(class_labels))}

        return jsonify({"predictions": result})

if __name__ == "__main__":
    app.run(debug=True)

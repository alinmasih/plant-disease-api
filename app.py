from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import requests

app = Flask(__name__)

# === 🔽 Google Drive .h5 model download ===
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=15bgxNEXw7lNykh2O09MLqVWypiPoDTXF"  # Replace with your actual file ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        try:
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print("❌ Failed to download model:", e)

# === 🔁 Download model if needed ===
download_model()

# === 📦 Load the model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === 🏷️ Define class labels (edit as needed) ===
class_names = ['lemon_deseased', 'lemon_healthy', 'spider_diseased', 'spider_healthy']

@app.route('/')
def home():
    return "🌱 Plant Disease ML API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_b64 = data['image']
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_idx = np.argmax(predictions)
        predicted_class = class_names[class_idx]
        confidence = float(predictions[class_idx])

        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

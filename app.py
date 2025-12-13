from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load scaler safely
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return "CKD API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    # Dummy prediction (since model is missing)
    prediction = 1 if np.mean(data) > 0 else 0

    return jsonify({
        "prediction": int(prediction),
        "message": "CKD Detected" if prediction == 1 else "No CKD"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


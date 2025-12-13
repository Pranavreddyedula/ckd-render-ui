from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read input features safely in correct order
    features = [float(x) for x in request.form.getlist("features[]")]

    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0]
    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        status = "Chronic Kidney Disease Detected"
        kidney_image = "ckd.png"
    else:
        status = "No Chronic Kidney Disease"
        kidney_image = "healthy.png"

    return render_template(
        "result.html",
        prediction=status,
        confidence=confidence,
        kidney_image=kidney_image
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("ckd_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(v) for v in request.form.values()]
    sample = np.array(values).reshape(1, -1)
    scaled = scaler.transform(sample).reshape(1, -1, 1)

    prob = model.predict(scaled)[0][0]
    confidence = round(prob * 100, 2)

    if prob >= 0.5:
        result = "⚠️ CKD DETECTED"
        kidney_image = "ckd_kidney.png"
    else:
        result = "✅ NO CKD"
        kidney_image = "healthy_kidney.png"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        kidney_image=kidney_image
    )

if __name__ == "__main__":
    app.run()

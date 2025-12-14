from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load trained model & scaler
model = tf.keras.models.load_model("ckd_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr',
    'bu','sc','sod','pot','hemo','pcv','wc','rc',
    'htn','dm','cad','appet','pe','ane'
]

CAUSES = [
    "Diabetes Mellitus",
    "High Blood Pressure",
    "Heart Disease",
    "Smoking",
    "Obesity",
    "Family History of CKD"
]

STAGES = {
    0: "Healthy Kidney",
    1: "Stage 1–2 CKD (Mild)",
    2: "Stage 3 CKD (Moderate)",
    3: "Stage 4–5 CKD (Severe)"
}

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[f]) for f in FEATURES]
    data = np.array(values).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prob = model.predict(data_scaled)[0][0]
    confidence = round(prob * 100, 2)

    if prob < 0.5:
        result = "✅ NO CKD"
        image = "healthy_kidney.png"
        stage = STAGES[0]
    else:
        result = "⚠️ CKD DETECTED"
        image = "ckd_kidney.png"
        stage = STAGES[min(3, int(prob * 4))]

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image=image,
        stage=stage,
        causes=CAUSES
    )

if __name__ == "__main__":
    app.run()

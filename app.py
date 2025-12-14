from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        values = [float(request.form[f]) for f in FEATURES]
        data = scaler.transform([values])
        pred = model.predict(data)[0]

        result = "NO CKD" if pred == 0 else "CKD DETECTED"
        image = "healthy_kidney.png" if pred == 0 else "ckd_kidney.png"

        return render_template(
            "result.html",
            prediction=result,
            image=image,
            confidence="High",
            causes=[
                "Diabetes",
                "High Blood Pressure",
                "Kidney Infections",
                "Heart Disease"
            ],
            stages=[
                "Stage 1: Mild damage",
                "Stage 2: Mild loss",
                "Stage 3: Moderate loss",
                "Stage 4: Severe loss",
                "Stage 5: Kidney failure"
            ]
        )

    return render_template("index.html", features=FEATURES)

if __name__ == "__main__":
    app.run()

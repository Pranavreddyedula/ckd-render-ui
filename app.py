from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler safely
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            data = scaler.transform([values])

            pred = model.predict(data)[0]
            confidence = "High" if pred == 1 else "Low"

            result = "CKD DETECTED" if pred == 1 else "NO CKD"
            image = "ckd_kidney.png" if pred == 1 else "healthy_kidney.png"

            return render_template(
                "result.html",
                prediction=result,
                confidence=confidence,
                image=image,
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
        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", features=FEATURES)

# IMPORTANT for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

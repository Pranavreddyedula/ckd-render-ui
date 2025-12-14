from flask import Flask, render_template, request
import pickle
import numpy as np

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

        prob = model.predict_proba(data)[0][1]
        pred = 1 if prob >= 0.5 else 0
        confidence = f"{round(prob*100, 2)} %"

        result = "NO CKD" if pred == 0 else "CKD DETECTED"
        image = "healthy_kidney.png" if pred == 0 else "ckd_kidney.png"

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

    return render_template("index.html", features=FEATURES)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained components
model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image = None

    if request.method == "POST":
        try:
            values = []
            for feature in FEATURES:
                val = request.form.get(feature)
                if val is None or val.strip() == "":
                    val = 0
                values.append(float(val))

            data = np.array(values).reshape(1, -1)
            data = imputer.transform(data)
            data = scaler.transform(data)

            result = model.predict(data)[0]

            if result == 1:
                prediction = "🩺 CKD Detected"
                image = "ckd_kidney.png"
            else:
                prediction = "✅ No CKD Detected"
                image = "healthy_kidney.png"

        except Exception as e:
            prediction = "❌ Invalid input values"
            image = None

    return render_template(
        "index.html",
        prediction=prediction,
        image=image
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

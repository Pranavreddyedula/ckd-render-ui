from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained components
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = []
            for feature in FEATURES:
                val = request.form.get(feature)
                if val == "" or val is None:
                    val = 0   # ✅ safe default
                values.append(float(val))

            data = np.array(values).reshape(1, -1)

            data = imputer.transform(data)
            data = scaler.transform(data)

            result = model.predict(data)[0]

            prediction = "🩺 CKD Detected" if result == 1 else "✅ No CKD Detected"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

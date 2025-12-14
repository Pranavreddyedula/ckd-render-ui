from flask import Flask, render_template, request
import numpy as np
import joblib

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
        values = []
        for feature in FEATURES:
            val = request.form.get(feature)
            values.append(float(val))

        data = np.array(values).reshape(1, -1)

        data = imputer.transform(data)
        data = scaler.transform(data)

        result = model.predict(data)[0]

        prediction = "🩺 CKD Detected" if result == 1 else "✅ No CKD Detected"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

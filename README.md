# 🩺 Chronic Kidney Disease Detection Web App

A machine learning–based web application that predicts whether a patient has **Chronic Kidney Disease (CKD)** based on medical input parameters.  
The application is built using **Flask** and **Scikit-learn**, and deployed on **Render**.

---

## 📌 Project Overview

Chronic Kidney Disease is a serious health condition that requires early detection.  
This project uses a trained Machine Learning model to analyze patient health attributes and predict the presence of CKD with confidence percentage.

The system provides:
- Easy-to-use web interface
- Real-time prediction
- Confidence score
- Visual result (healthy vs CKD kidney image)

---

## 🚀 Features

- ✅ 24 medical parameters as input
- 🧠 Machine Learning model (Logistic Regression)
- 📊 Prediction confidence score
- 🖼️ Visual kidney status image
- 🌐 Web-based interface using Flask
- ☁️ Deployed on Render (Free Tier)

---

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **HTML / CSS**
- **Gunicorn**
- **Render Cloud Platform**

---

## 📂 Project Structure
CKD-Detection/
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── static/
│ ├── style.css
│ ├── ckd.png
│ └── healthy.png
│
├── app.py
├── model.py
├── model.pkl
├── scaler.pkl
├── ckdisease.csv
├── requirements.txt
├── render.yaml
└── README.md

---

## ⚙️ How It Works

1. User enters 24 medical values on the homepage.
2. Data is scaled using a trained `StandardScaler`.
3. Machine Learning model predicts CKD or Non-CKD.
4. Confidence score is calculated using prediction probabilities.
5. Result page displays:
   - Disease status
   - Confidence percentage
   - Corresponding kidney image

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the app
python app.py

3️⃣ Open browser
http://127.0.0.1:10000

☁️ Deployment (Render)

Platform: Render

Plan: Free

Start Command:

gunicorn app:app


The application automatically binds to the assigned port using environment variables.

📊 Dataset

Source: CKD dataset (CSV format)

File: ckdisease.csv

Target variable:

ckd → 1

notckd → 0

🎓 Academic Use

Suitable for Final Year CSE / IT Projects

Easy to explain in Viva

Covers:

Machine Learning

Web Development

Cloud Deployment

🔮 Future Enhancements

Add Deep Learning model

Feature importance visualization

User authentication

PDF medical report generation

Doctor recommendation system

👤 Author

Edula Sai Pranav Reddy
CSE Student
GitHub: https://github.com/Pranavreddyedula

📜 License

This project is for educational purposes only.

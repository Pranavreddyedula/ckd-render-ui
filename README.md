# 🩺 Chronic Kidney Disease Detection (Deep Learning)

A deep learning–based web application to detect CKD using medical parameters.

## Features
- CNN/DNN-based CKD detection
- Prediction confidence (%)
- Disease stages
- Possible causes
- Hospital-style UI
- Deployed on Render

## Run Locally
```bash
pip install -r requirements.txt
python app.py
Deployment

Hosted using Render cloud platform.

---
✅ HOW TO FILL THE CKD FORM (SIMPLE & SAFE)
🧠 Important rule

Enter only numbers

For Yes/No fields → use 0 or 1

For medical values → use approximate normal values (demo is OK)

📌 MEANING + SAMPLE VALUES (COPY–PASTE READY)
🔢 Basic details
age   : 45
bp    : 80
sg    : 1.02
al    : 1
su    : 0

🧪 Urine & blood tests
rbc   : 1     (1 = normal, 0 = abnormal)
pc    : 1
pcc   : 0
ba    : 0
bgr   : 120
bu    : 36
sc    : 1.2
sod   : 135
pot   : 4.5
hemo  : 13
pcv   : 40
wc    : 8000
rc    : 4.8

❤️ Medical conditions (0 = No, 1 = Yes)
htn   : 0
dm    : 0
cad   : 0

🍽 Appetite & symptoms
appet : 1     (1 = good, 0 = poor)
pe    : 0
ane   : 0

🧾 QUICK CHEAT SHEET (VERY IMPORTANT)
Field	Meaning	Value
rbc, pc	normal/abnormal	1 = normal, 0 = abnormal
pcc, ba	present/absent	1 = present, 0 = absent
htn, dm, cad	disease	1 = yes, 0 = no
appet	appetite	1 = good, 0 = poor
pe, ane	symptoms	1 = yes, 0 = no
▶️ AFTER FILLING

Click Predict

You’ll get:

✅ No CKD Detected
OR

🩺 CKD Detected

(with kidney image & accuracy graph)

# 🚀 FINAL RESULT (WHAT YOUR PROJECT DOES)

✅ Live CKD prediction  
✅ Shows **Healthy / CKD kidney image**    
✅ Clean UI  
✅ Render deploy works  
 


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("ckdisease.csv")

# Remove unwanted index column if exists
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Replace missing values
df.replace("?", np.nan, inplace=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Binary encoding
binary_map = {
    "yes": 1, "no": 0,
    "normal": 1, "abnormal": 0,
    "present": 1, "notpresent": 0,
    "good": 1, "poor": 0,
    "ckd": 1, "notckd": 0
}

for col in df.columns:
    df[col] = df[col].map(binary_map).fillna(df[col])

# Convert all to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# 🔥 EXACT 24 FEATURES (NO MORE, NO LESS)
FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

# Select inputs and target
X = df.loc[:, FEATURES]
y = df['classification']

# Remove rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save trained objects
joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ CKD model trained with EXACTLY 24 features!")

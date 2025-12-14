import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("ckdisease.csv")

# Replace missing values
df.replace("?", np.nan, inplace=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Binary encoding
binary_map = {
    "yes":1, "no":0,
    "normal":1, "abnormal":0,
    "present":1, "notpresent":0,
    "good":1, "poor":0,
    "ckd":1, "notckd":0
}

for col in df.columns:
    df[col] = df[col].map(binary_map).fillna(df[col])

df = df.apply(pd.to_numeric, errors="coerce")

# ✅ DEFINE EXACT 24 FEATURES
FEATURES = [
 'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
 'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

X = df[FEATURES]
y = df['classification']

# Remove rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]

# Imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save models
joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ CKD Model trained with 24 features and saved successfully!")

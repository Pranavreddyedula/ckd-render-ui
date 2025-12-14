import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("ckdisease.csv")

# 🔥 Normalize column names (THIS IS THE KEY FIX)
df.columns = df.columns.str.strip().str.lower()

# Remove unnamed / index columns
df = df.loc[:, ~df.columns.str.contains("^unnamed")]

# Replace missing values
df.replace("?", np.nan, inplace=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Encode categorical values
binary_map = {
    "yes": 1, "no": 0,
    "normal": 1, "abnormal": 0,
    "present": 1, "notpresent": 0,
    "good": 1, "poor": 0,
    "ckd": 1, "notckd": 0
}

for col in df.columns:
    df[col] = df[col].map(binary_map).fillna(df[col])

# Convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# ✅ EXACT 24 FEATURES (LOWERCASE, STRIPPED)
FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

# 🔥 FORCE ONLY THESE COLUMNS
X = df[FEATURES].copy()
y = df['classification']

# Remove rows with missing target
mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]

# 🚨 HARD ASSERT (CANNOT FAIL SILENTLY)
assert X.shape[1] == 24, f"Feature count is {X.shape[1]}, expected 24"

# Impute
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save
joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ CKD model trained with EXACTLY 24 features")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Plot accuracy graph
plt.figure()
plt.bar(["Training Accuracy", "Testing Accuracy"],
        [train_acc, test_acc])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("CKD Model Accuracy")
plt.savefig("static/accuracy.png")
plt.close()

print("📊 Accuracy graph saved as static/accuracy.png")



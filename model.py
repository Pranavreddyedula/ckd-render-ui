import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("ckdisease.csv")

# Drop unwanted columns
df = df.dropna()

X = df.drop("classification", axis=1)
y = df["classification"].map({"ckd": 1, "notckd": 0})

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully")

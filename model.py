import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load dataset
df = pd.read_csv("ckdisease.csv")
df.replace("?", np.nan, inplace=True)

binary_map = {
    "normal":1, "abnormal":0,
    "present":1, "notpresent":0,
    "yes":1, "no":0,
    "good":1, "poor":0,
    "ckd":1, "notckd":0
}

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].map(binary_map)

df.dropna(inplace=True)

X = df.drop("classification", axis=1)
y = df["classification"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model = Sequential([
    Conv1D(32, 2, activation="relu", input_shape=(X_train.shape[1],1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=16)

# ✅ THESE CREATE VALID FILES
model.save("ckd_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("SUCCESS: Model and scaler saved")

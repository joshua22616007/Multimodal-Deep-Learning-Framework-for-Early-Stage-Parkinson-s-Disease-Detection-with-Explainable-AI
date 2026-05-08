import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
df = pd.read_csv("dataset/audio.csv")
print(df.columns)
# Drop non-feature columns (adjust if needed)
if "name" in df.columns:
    df = df.drop(columns=["name"])
    
    
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "gender" in df.columns:
    df = df.drop(columns=["gender"])

# Target column
y = df["class"]
X = df.drop(columns=["class"])

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save
model.save("models/audio_model.h5")

print("✅ Model trained and saved!")
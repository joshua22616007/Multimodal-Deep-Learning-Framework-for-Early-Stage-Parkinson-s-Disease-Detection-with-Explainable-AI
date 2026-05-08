import numpy as np
import librosa
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===== FEATURE EXTRACTION =====
def extract_features(file):
    y, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ===== LOAD DATA =====
X = []
y = []

dataset_path = "audio_dataset"

for label in ["healthy", "parkinson"]:
    path = os.path.join(dataset_path, label)
    class_label = 0 if label == "healthy" else 1

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        try:
            feat = extract_features(file_path)
            X.append(feat)
            y.append(class_label)
        except:
            continue

X = np.array(X)
y = np.array(y)

# ===== MODEL =====
model = Sequential([
    Dense(128, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ===== TRAIN =====
model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)

# ===== SAVE =====
model.save("models/audio_model.h5")

print("✅ Audio model trained and saved!")
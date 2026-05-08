import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

def load_data(folder):
    X, y = [], []

    for label in ["healthy", "parkinson"]:
        path = os.path.join(folder, label)

        if not os.path.exists(path):
            print("Folder missing:", path)
            continue

        class_label = 0 if label == "healthy" else 1

        for file in os.listdir(path):
            img_path = os.path.join(path, file)

            try:
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                X.append(img)
                y.append(class_label)
            except:
                continue

    return np.array(X), np.array(y)

# ===== LOAD DATA =====
X, y = load_data("handwriting_dataset")

# ===== CHECK DATA =====
print("Data loaded:", len(X))

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# ===== MODEL =====
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ===== TRAIN =====
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# ===== SAVE =====
model.save("models/handwriting_model.h5")

print("✅ Handwriting model trained & saved!")
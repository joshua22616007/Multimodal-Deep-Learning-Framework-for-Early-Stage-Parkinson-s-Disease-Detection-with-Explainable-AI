import numpy as np
import librosa
import joblib
import os
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


from tensorflow.keras.models import load_model
from utils.counterfactual import generate_counterfactual

# 🔥 NEW: for recording
import sounddevice as sd
from scipy.io.wavfile import write


# 🔥 Load trained model + scaler + feature names
model = load_model("models/audio_model.h5")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# 🔥 CREATE RECORDS FOLDER IF NOT EXISTS
if not os.path.exists("records"):
    os.makedirs("records")


# 🎤 🔥 REAL-TIME AUDIO RECORD FUNCTION
recording_data = []
is_recording = False


def start_recording(fs=22050):
    global recording_data, is_recording
    recording_data = []
    is_recording = True

    def callback(indata, frames, time_info, status):
        if is_recording:
            recording_data.append(indata.copy())

    stream = sd.InputStream(callback=callback, channels=1, samplerate=fs)
    stream.start()

    return stream


def stop_recording(stream, fs=22050):
    global is_recording
    is_recording = False

    stream.stop()
    stream.close()

    audio = np.concatenate(recording_data, axis=0)

    if not os.path.exists("records"):
        os.makedirs("records")

    filename = f"records/audio_{int(time.time())}.wav"
    write(filename, fs, audio)

    return filename, audio

def noise_filter(audio):
    # simple normalization filter
    audio = audio / np.max(np.abs(audio))
    return audio

def show_waveform(audio):
    plt.figure()
    plt.plot(audio)
    plt.title("Live Audio Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
    
def play_audio(audio, fs=22050):
    sd.play(audio, fs)
    sd.wait()


# 🔥 FEATURE EXTRACTION
def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    features = []

    # Core features
    features.append(np.mean(librosa.feature.rms(y=y)))  # Energy
    features.append(np.mean(librosa.yin(y, fmin=50, fmax=300)))  # Pitch
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # Match dataset feature size
    while len(features) < len(feature_names):
        features.append(0)

    return np.array(features[:len(feature_names)]).reshape(1, -1)


# 🔥 MAIN PREDICTION FUNCTION (FILE-BASED)
def predict_audio(file):
    features = extract_features(file)

# 🔥 NORMALIZE INPUT
    features = features / (np.max(np.abs(features)) + 1e-6)

    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0][0]

# 🔥 CLAMP OUTPUT
    pred = max(0.1, min(pred, 0.9))

    # 🔥 CONFIDENCE
    confidence = abs(pred - 0.5) * 2
    confidence = min(confidence, 0.95)

    explanation = {
        feature_names[i]: float(features[0][i])
        for i in range(len(feature_names))
    }

    counterfactual = generate_counterfactual(features, feature_names)

    return float(pred), explanation, counterfactual, float(confidence)


# 🔥 NEW: REAL-TIME PREDICTION (DIRECT USE)
def predict_audio_live(duration=5):
    file = record_audio(duration)
    return predict_audio(file)
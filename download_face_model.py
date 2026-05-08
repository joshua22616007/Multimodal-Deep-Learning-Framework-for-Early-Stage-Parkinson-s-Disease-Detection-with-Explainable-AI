import urllib.request
import os

os.makedirs("models/face_dnn", exist_ok=True)

print("Downloading deploy.prototxt...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "models/face_dnn/deploy.prototxt"
)

print("Downloading model...")
urllib.request.urlretrieve(
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "models/face_dnn/res10_300x300_ssd_iter_140000.caffemodel"
)

print("✅ Download complete!")
import numpy as np
import json

import onnxruntime as ort

from io import BytesIO
from urllib import request
from PIL import Image

onnx_model_path = "hair_classifier_empty.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def preprocess_pytorch(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)


def predict(url):
    # 1. Download and resize the image
    img = download_image(url)
    img = prepare_image(img, (200, 200))  # 200x200 is the right target size

    # 2. Convert PIL → numpy → add batch dim
    X = np.array(img).astype("float32")   # shape: (200, 200, 3)
    X = np.expand_dims(X, axis=0)         # shape: (1, 200, 200, 3)

    # 3. Preprocess like PyTorch / ONNX expects
    X = preprocess_pytorch(X)             # shape: (1, 3, 200, 200)

    # 4. Run ONNX inference
    result = session.run([output_name], {input_name: X})
    # result is a list with one array inside → [array([[0.09]], dtype=float32)]

    score = float(result[0][0][0])        # extract the scalar

    return score


def lambda_handler(event, context):
    """
    Lambda entrypoint.
    Expects event like: {"url": "https://..."}
    """
    url = event.get("url")

    if not url:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "url is required"})
        }

    try:
        score = predict(url)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    # You can return just score, but wrapping in JSON is cleaner
    return {
        "statusCode": 200,
        "body": json.dumps({"score": score})
    }

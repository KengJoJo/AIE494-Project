import onnxruntime as ort
from PIL import Image
import requests
from transformers import AutoImageProcessor
import numpy as np
import os
import json

# 1. Load Data
model_dir = "models/original"
onnx_path = "models/onnx/model.onnx"
quant_path = "models/quantized/model_quantized.onnx"

with open(os.path.join(model_dir, "id2label.json"), "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

processor = AutoImageProcessor.from_pretrained(model_dir)

# 2. Get Cat Image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
input_array = inputs["pixel_values"].numpy().astype(np.float32)

def run_onnx(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    top_idx = probs.argmax()
    return id2label[top_idx], probs[top_idx]

print("Comparing Model Versions on REAL CAT:")
try:
    label, score = run_onnx(onnx_path)
    print(f"  [ONNX Normal]    -> {label} ({score:.4f})")
except Exception as e:
    print(f"  [ONNX Normal]    -> Error: {e}")

try:
    label, score = run_onnx(quant_path)
    print(f"  [Quantized ONNX] -> {label} ({score:.4f})")
except Exception as e:
    print(f"  [Quantized ONNX] -> Error: {e}")

import os
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor
import numpy as np

model_dir = "models/original"
onnx_path = "models/quantized/model_quantized.onnx"
processor = AutoImageProcessor.from_pretrained(model_dir)

# Create a clear red square
img = Image.new("RGB", (224, 224), color=(255, 0, 0))
inputs = processor(images=img, return_tensors="np")
input_array = inputs["pixel_values"].astype(np.float32)

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_array})
logits = outputs[0][0]

# Softmax
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / exp_logits.sum()
top_indices = probs.argsort()[::-1][:5]

# Load labels
import json
with open(os.path.join(model_dir, "id2label.json"), "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

print("Top 5 Predictions (Quantized ONNX):")
for idx in top_indices:
    label = id2label[idx]
    score = probs[idx]
    print(f"  {label}: {score:.4f}")

import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import os

# 1. Load Original Model & Processor
model_id = "models/original"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

# 2. Get a REAL cat image from a reliable source
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Process with Processor
inputs = processor(images=image, return_tensors="pt")

# 4. Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_prob, top_idx = torch.topk(probs, 5)

print(f"--- Original PyTorch Prediction for REAL CAT ---")
for i in range(5):
    idx = top_idx[0][i].item()
    label = model.config.id2label[idx]
    score = top_prob[0][i].item()
    print(f"  {label}: {score:.4f}")

# 5. Check the values of inputs
pixel_values = inputs["pixel_values"].numpy()
print(f"\nPixel values range: min={pixel_values.min():.4f}, max={pixel_values.max():.4f}, mean={pixel_values.mean():.4f}")

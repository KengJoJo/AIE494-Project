import os
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import numpy as np

model_dir = "models/original"
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForImageClassification.from_pretrained(model_dir)

# Create a clear red square (should be identified as something or at least consistent)
img = Image.new("RGB", (224, 224), color=(255, 0, 0))
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_prob, top_idx = torch.topk(probs, 5)

print("Top 5 Predictions (PyTorch Original):")
for i in range(5):
    idx = top_idx[0][i].item()
    label = model.config.id2label[idx]
    score = top_prob[0][i].item()
    print(f"  {label}: {score:.4f}")

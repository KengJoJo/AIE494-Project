import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import json

def load_labels(model_dir):
    label_path = os.path.join(model_dir, "id2label.json")
    with open(label_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
    img_data = img_data / 255.0
    # Simple normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    return img_data[np.newaxis, :]

def benchmark(model_path, image_path, runs=50):
    print(f"Benchmarking {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_data = preprocess_image(image_path)
    
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: input_data})
    
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg = np.mean(latencies)
    p95 = np.percentile(latencies, 95)
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Avg: {avg:.2f}ms | P95: {p95:.2f}ms | Size: {size:.2f}MB")
    return {"avg": avg, "p95": p95, "size": size}

def main():
    image_path = "test_cat.jpg"
    models = {
        "ONNX (Standard)": "models/onnx/model.onnx",
        "Quantized ONNX": "models/quantized/model_quantized.onnx"
    }
    
    results = []
    for name, path in models.items():
        if os.path.exists(path):
            res = benchmark(path, image_path)
            results.append(f"| {name} | {res['size']:.2f} | {res['avg']:.2f} | {res['p95']:.2f} |")
    
    print("\n### Benchmark Summary")
    print("| Model | Size (MB) | Avg Latency (ms) | P95 Latency (ms) |")
    print("|---|---|---|---|")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()

"""
Benchmark script - compares Original PyTorch vs ONNX vs Quantized ONNX models.

Measures:
    - Model size (MB)
    - Average latency (ms)
    - P50, P95, Min, Max latency

Usage:
    python scripts/benchmark.py --image path/to/image.jpg
    python scripts/benchmark.py --image path/to/image.jpg --warmup 50 --runs 100
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_model_size_mb(path: str) -> float:
    """Get file size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    # For PyTorch models stored as directories, sum all .bin/.safetensors files
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith((".bin", ".safetensors", ".pt", ".pth")):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def benchmark_pytorch(image_bytes: bytes, model_dir: str, warmup: int, runs: int):
    """Benchmark original PyTorch model."""
    from transformers import AutoModelForImageClassification
    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.inference import preprocess_image

    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()

    input_array = preprocess_image(image_bytes)
    input_tensor = torch.from_numpy(input_array)

    # Warm up
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            model(input_tensor)

    # Measure
    latencies = []
    print(f"  Benchmarking ({runs} iterations)...")
    for _ in tqdm(range(runs), desc="  PyTorch"):
        start = time.perf_counter()
        with torch.no_grad():
            model(input_tensor)
        latencies.append((time.perf_counter() - start) * 1000)

    size_mb = get_model_size_mb(model_dir)
    return size_mb, latencies


def benchmark_onnx(image_bytes: bytes, model_path: str, warmup: int, runs: int):
    """Benchmark ONNX model."""
    import onnxruntime as ort

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.inference import preprocess_image

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_array = preprocess_image(image_bytes)

    # Warm up
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    # Measure
    latencies = []
    print(f"  Benchmarking ({runs} iterations)...")
    for _ in tqdm(range(runs), desc="  ONNX"):
        start = time.perf_counter()
        session.run(None, {input_name: input_array})
        latencies.append((time.perf_counter() - start) * 1000)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return size_mb, latencies


def compute_stats(latencies: list) -> dict:
    """Compute latency statistics."""
    arr = np.array(latencies)
    return {
        "avg_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark image classification models")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warm-up iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of measured iterations")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    image_path = os.path.join(project_root, args.image)
    if not os.path.exists(image_path) and not os.path.exists(args.image):
        print("Please provide a valid image path with --image path/to/image.jpg")
        sys.exit(1)
        
    # Prefer absolute path if provided, else relative to project root
    actual_path = args.image if os.path.exists(args.image) else image_path

    with open(actual_path, "rb") as f:
        image_bytes = f.read()

    results = []
    models_dir = os.path.join(project_root, "models")

    # --- 1. Original PyTorch ---
    original_dir = os.path.join(models_dir, "original")
    if os.path.isdir(original_dir):
        print("\n[1/3] Benchmarking Original PyTorch model...")
        size_mb, latencies = benchmark_pytorch(image_bytes, original_dir, args.warmup, args.runs)
        stats = compute_stats(latencies)
        results.append({"Model Type": "Original (PyTorch)", "Size (MB)": round(size_mb, 2), **stats})
        print(f"  -> Avg: {stats['avg_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms")
    else:
        print(f"\n[1/3] Skipping PyTorch -- not found at {original_dir}")

    # --- 2. ONNX ---
    onnx_path = os.path.join(models_dir, "onnx", "model.onnx")
    if os.path.exists(onnx_path):
        print("\n[2/3] Benchmarking ONNX model...")
        size_mb, latencies = benchmark_onnx(image_bytes, onnx_path, args.warmup, args.runs)
        stats = compute_stats(latencies)
        results.append({"Model Type": "ONNX", "Size (MB)": round(size_mb, 2), **stats})
        print(f"  -> Avg: {stats['avg_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms")
    else:
        print(f"\n[2/3] Skipping ONNX -- not found at {onnx_path}")

    # --- 3. Quantized ONNX ---
    quant_path = os.path.join(models_dir, "quantized", "model_quantized.onnx")
    if os.path.exists(quant_path):
        print("\n[3/3] Benchmarking Quantized ONNX model...")
        size_mb, latencies = benchmark_onnx(image_bytes, quant_path, args.warmup, args.runs)
        stats = compute_stats(latencies)
        results.append({"Model Type": "Quantized ONNX", "Size (MB)": round(size_mb, 2), **stats})
        print(f"  -> Avg: {stats['avg_ms']:.2f} ms | P95: {stats['p95_ms']:.2f} ms")
    else:
        print(f"\n[3/3] Skipping Quantized ONNX -- not found at {quant_path}")

    if not results:
        print("\nNo models found. Run model preparation scripts first.")
        sys.exit(1)

    # --- Save results ---
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.columns = ["Model Type", "Size (MB)", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)"]

    # CSV
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV saved to: {csv_path}")

    # Markdown
    md_path = os.path.join(results_dir, "benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"**Image:** {args.image}  \n")
        f.write(f"**Warm-up:** {args.warmup} iterations  \n")
        f.write(f"**Measured:** {args.runs} iterations  \n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"[OK] Markdown saved to: {md_path}")

    # Print table
    print("\n" + "=" * 80)
    print(df.to_markdown(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

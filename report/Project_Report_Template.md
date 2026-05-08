# Project Report: High-Throughput Image Classification Service

> **Course:** AIE494  
> **Student Name:** [Your Name]  
> **Student ID:** [Your Student ID]  
> **Date:** [Submission Date]

---

## 1. Project Overview

This project implements a **High-Throughput Image Classification API** using FastAPI, ONNX Runtime, and dynamic quantization. The system accepts uploaded images, classifies them using a MobileNetV2 model, and returns top-K predictions with confidence scores.

**Key Features:**
- Real-time image classification via REST API
- ONNX conversion and dynamic quantization for optimized CPU inference
- Concurrent request handling with ProcessPoolExecutor
- Docker packaging for deployment
- CI/CD pipeline with GitHub Actions
- Load testing with JMeter

---

## 2. Model Selection

| Attribute       | Value                              |
|-----------------|-------------------------------------|
| Model           | `google/mobilenet_v2_1.0_224`      |
| Source           | Hugging Face Transformers          |
| Task             | Image Classification (ImageNet-1K) |
| Input Size       | 224 × 224 × 3                      |
| Number of Classes| 1,000                              |
| Parameters       | ~3.4M                              |

**Why MobileNetV2?**
- Lightweight architecture designed for mobile and edge deployment
- Fast CPU inference suitable for real-time API serving
- Good accuracy-to-speed tradeoff for production use
- Widely supported by ONNX Runtime

---

## 3. Purpose of the Model

The model classifies input images into one of 1,000 ImageNet categories. This demonstrates:
- How to serve ML models via REST APIs
- Model optimization techniques (ONNX, quantization)
- Production-grade error handling and validation
- Scalable concurrent inference architecture

---

## 4. System Architecture

See [architecture.md](architecture.md) for the detailed system diagram.

**Components:**
1. **FastAPI Server** — Handles HTTP requests, validation, and response formatting
2. **Validation Layer** — Checks file type, size, and image integrity
3. **ProcessPoolExecutor** — Dispatches CPU-bound inference to worker processes
4. **ONNX Runtime Worker** — Runs the quantized model in separate processes
5. **Docker Container** — Packages the entire application
6. **GitHub Actions** — Automates testing and deployment

---

## 5. API Design

### Endpoints

| Method | Path      | Description                    |
|--------|-----------|--------------------------------|
| GET    | `/`       | Service status                 |
| GET    | `/health` | Health check with model status |
| POST   | `/predict`| Image classification           |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/image.jpg;type=image/jpeg"
```

### Example Response

```json
{
  "filename": "image.jpg",
  "content_type": "image/jpeg",
  "model_type": "quantized_onnx",
  "latency_ms": 35.42,
  "predictions": [
    {"label": "spider web", "score": 0.9234},
    {"label": "web site", "score": 0.0312},
    {"label": "barn spider", "score": 0.0156}
  ]
}
```

---

## 6. Error Handling and Data Validation

| Scenario              | HTTP Status | Response                          |
|-----------------------|-------------|-----------------------------------|
| Valid image           | 200         | JSON predictions                  |
| Invalid file type     | 400         | "Invalid file type"               |
| Corrupted image       | 400         | "Not a valid image or corrupted"  |
| File too large (>5MB) | 413         | "Exceeds maximum allowed size"    |
| Missing file field    | 422         | "Field required"                  |
| Server error          | 500         | "Inference error: ..."            |

**Validation checks:**
- Content type must be `image/jpeg`, `image/png`, or `image/webp`
- File size must be ≤ 5 MB (configurable)
- Image must pass Pillow `verify()` integrity check

---

## 7. Model Optimization Results

> **Instructions:** Run `python scripts/benchmark.py --image path/to/image.jpg` and paste the results below.

| Model Type       | Size (MB) | Avg (ms) | P50 (ms) | P95 (ms) | Notes                   |
|------------------|-----------|----------|----------|----------|-------------------------|
| Original (PyTorch) | ___.__ | ___.__ | ___.__ | ___.__ | Baseline                |
| ONNX             | ___.__ | ___.__ | ___.__ | ___.__ | Converted from PyTorch  |
| Quantized ONNX   | ___.__ | ___.__ | ___.__ | ___.__ | INT8 dynamic quantization |

**Analysis:**
- [Describe size reduction from Original → ONNX → Quantized]
- [Describe latency improvements]
- [Note any accuracy tradeoffs observed]

---

## 8. JMeter Load Testing

### 8.1 Local Docker Results

> Run: `jmeter -n -t jmeter/image_classification_load_test.jmx -l results/local_results.jtl -e -o results/local_dashboard`

| Metric          | Value     |
|-----------------|-----------|
| Total Requests  | 500       |
| Throughput (TPS)| ___.__ /s |
| Avg Response Time | ___ ms  |
| P95 Latency     | ___ ms    |
| Error Rate      | ____%     |

### 8.2 Hugging Face Cloud Results

> Run with `-JHOST=<your-space>.hf.space -JPORT=443 -JPROTOCOL=https`

| Metric          | Value     |
|-----------------|-----------|
| Total Requests  | 500       |
| Throughput (TPS)| ___.__ /s |
| Avg Response Time | ___ ms  |
| P95 Latency     | ___ ms    |
| Error Rate      | ____%     |

### 8.3 Bottleneck Analysis

- [Describe where bottlenecks occur: CPU inference, network, memory, etc.]
- [Compare local vs cloud performance]
- [Discuss impact of Docker CPU/memory limits]
- [Suggestions for scaling: more workers, GPU, caching, etc.]

---

## 9. CI/CD Pipeline

**Pipeline stages:**
1. **Checkout** — Pull latest code
2. **Setup Python** — Install Python 3.11
3. **Install Dependencies** — `pip install -r requirements.txt`
4. **Run Tests** — `pytest -q`
5. **Deploy** — Upload to Hugging Face Spaces (main branch only)

**Secrets required:**
- `HF_TOKEN` — Hugging Face API token
- `HF_SPACE_ID` — Target Space ID (e.g., `username/image-classification`)

---

## 10. Deployment

### Hugging Face Spaces

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space)
   - SDK: Docker
   - Hardware: CPU Basic (free)
2. Set GitHub Secrets:
   - `HF_TOKEN`: your HF write token
   - `HF_SPACE_ID`: `your-username/your-space-name`
3. Push to `main` branch → CI/CD auto-deploys

### Live URL

```
https://<your-username>-<your-space-name>.hf.space
```

---

## 11. Problems and Solutions

| # | Problem | Solution |
|---|---------|----------|
| 1 | [Describe problem] | [Describe solution] |
| 2 | [Describe problem] | [Describe solution] |
| 3 | [Describe problem] | [Describe solution] |

---

## 12. Conclusion

[Summarize what was achieved, key learnings, and potential future improvements such as:]
- GPU inference support
- Model caching and warm-up strategies
- Additional model architectures
- Real-time monitoring and alerting
- Auto-scaling based on load

---

## Appendix

### A. Project Repository Structure

```
├── app/                    # FastAPI application
├── scripts/                # Model preparation scripts
├── tests/                  # Pytest test suite
├── jmeter/                 # Load testing artifacts
├── report/                 # This report
├── models/                 # Model artifacts (gitignored)
├── results/                # Benchmark & test results
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### B. Technology Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| Web Framework   | FastAPI                       |
| ML Framework    | PyTorch + Transformers        |
| Inference Engine| ONNX Runtime                  |
| Optimization    | Dynamic Quantization (INT8)   |
| Concurrency     | ProcessPoolExecutor + asyncio |
| Containerization| Docker                        |
| CI/CD           | GitHub Actions                |
| Load Testing    | Apache JMeter                 |

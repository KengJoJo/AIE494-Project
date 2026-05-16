---
title: AIE494 Image Classification API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AIE494 Image Classification API

High-throughput image classification service built with FastAPI, MobileNetV2,
ONNX Runtime, Docker, GitHub Actions, and JMeter.

## Team Members

1. นายธนนท์ จิตรพรหม 1650904194
2. นายเจตนิพัทธ อินทรีย์ 1650901471
3. นายนิธิกุล แก้วไพฑูรย์ 1650903808

## Project Links

- GitHub repository: https://github.com/KengJoJo/AIE494-Project
- Hugging Face Space page: https://huggingface.co/spaces/KengJoJo/AIE494-Project
- Cloud API URL: https://kengjojo-aie494-project.hf.space

The production API uses the standard ONNX model (`MODEL_TYPE=onnx`) because the
local benchmark showed the best latency and the most reliable predictions. A
quantized ONNX model is also included as an optimization experiment.

## Features

- REST API for image classification
- MobileNetV2 ImageNet-1K model from Hugging Face
- ONNX Runtime CPU inference
- Image upload validation for file type, file size, and image integrity
- ProcessPoolExecutor for CPU-bound inference work
- Docker and Docker Compose support
- GitHub Actions CI/CD pipeline
- JMeter load-test plan
- Postman collection and cURL examples

## Project Structure

```text
app/                         FastAPI application code
models/                      Model artifacts
scripts/                     Model preparation and benchmark scripts
tests/                       Pytest test suite
jmeter/                      JMeter load-test plan
postman/                     Postman API collection
report/                      Project report source files
results/                     Benchmark and generated testing outputs
.github/workflows/ci-cd.yml  CI/CD workflow
Dockerfile                   Production container image
docker-compose.yml           Local Docker Compose setup
```

## Requirements

- Python 3.10 or 3.11
- Docker Desktop, optional but recommended
- Apache JMeter 5.6 or newer for load testing

Install dependencies:

```bash
pip install -r requirements.txt
```

If `python` points to the Microsoft Store alias on Windows, use the full Python
path or configure your PATH before running the scripts.

## Prepare Models

The repository already contains model artifacts. If you need to regenerate them,
run:

```bash
python scripts/download_model.py
python scripts/export_onnx.py
python scripts/quantize_onnx.py
```

## Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Run with Docker

```bash
docker compose up --build
```

The service will be available at:

```text
http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service status |
| GET | `/health` | Health check and active model status |
| POST | `/predict` | Upload an image and return top-K predictions |

## cURL Examples

Local API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@test_cat.jpg;type=image/jpeg"
```

Hugging Face Spaces API:

```bash
curl -X POST "https://kengjojo-aie494-project.hf.space/predict" \
  -H "accept: application/json" \
  -F "file=@test_cat.jpg;type=image/jpeg"
```

Example response:

```json
{
  "filename": "test_cat.jpg",
  "content_type": "image/jpeg",
  "model_type": "onnx",
  "latency_ms": 12.34,
  "predictions": [
    {
      "label": "tabby, tabby cat",
      "score": 0.7842
    }
  ]
}
```

## Postman

Import this file into Postman:

```text
postman/image-classification-api.postman_collection.json
```

Set collection variables:

| Variable | Example |
| --- | --- |
| `baseUrl` | `http://localhost:8000` or `https://kengjojo-aie494-project.hf.space` |
| `imagePath` | `C:/Work/AIE494/ProjectAIE494/test_cat.jpg` |

## Test

```bash
pytest -q
```

Current local result:

```text
17 passed
```

## Benchmark

```bash
python scripts/benchmark.py --image test_cat.jpg --warmup 20 --runs 50
```

Current benchmark summary:

| Model Type | Size (MB) | Avg (ms) | P50 (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original PyTorch | 13.54 | 17.74 | 17.32 | 22.22 |
| ONNX | 0.33 | 2.23 | 2.19 | 2.60 |
| Quantized ONNX | 3.60 | 20.10 | 19.36 | 22.25 |

Full benchmark output:

```text
results/benchmark_results.md
results/benchmark_results.csv
```

## JMeter Load Test

Start the API first, then run:

```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=http -JHOST=localhost -JPORT=8000 \
  -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" \
  -l results/local_results.jtl \
  -e -o results/local_dashboard
```

Cloud test:

```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=https -JHOST=kengjojo-aie494-project.hf.space -JPORT=443 \
  -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" \
  -l results/cloud_results.jtl \
  -e -o results/cloud_dashboard
```

JMeter plan details:

- 50 concurrent users
- 10 second ramp-up
- 10 loops per user
- 500 total requests
- POST `/predict` with multipart image upload
- HTTP 200 and JSON predictions assertions

## CI/CD

Workflow file:

```text
.github/workflows/ci-cd.yml
```

Pipeline stages:

1. Checkout repository
2. Set up Python 3.11
3. Install dependencies
4. Run pytest
5. Prepare model artifacts
6. Deploy to Hugging Face Spaces on pushes to `main`

Required GitHub secrets:

| Secret | Description |
| --- | --- |
| `HF_TOKEN` | Hugging Face write token |
| `HF_SPACE_ID` | Target Space ID, for example `username/image-classification-api` |

## Submission Checklist

- `report/Project_Report_Template.md` exported as PDF
- GitHub repository URL
- `.github/workflows/ci-cd.yml`
- `jmeter/image_classification_load_test.jmx`
- Generated JMeter HTML dashboard
- `postman/image-classification-api.postman_collection.json`
- Cloud cURL command for `/predict`

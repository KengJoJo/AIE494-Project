# JMeter Load Testing Guide

This folder contains the JMeter test plan for the image classification API.

## Prerequisites

- Apache JMeter 5.6 or newer
- The API running locally or on Hugging Face Spaces
- A local test image, for example `test_cat.jpg`

## Start Local API

Using Python:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Using Docker:

```bash
docker compose up --build
```

The local API should be available at:

```text
http://localhost:8000
```

## Run Local Load Test

```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=http -JHOST=localhost -JPORT=8000 \
  -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" \
  -l results/local_results.jtl \
  -e -o results/local_dashboard
```

Open the generated dashboard:

```text
results/local_dashboard/index.html
```

## Run Cloud Load Test

The Hugging Face Spaces hostname for this project is
`kengjojo-aie494-project.hf.space`.

```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=https -JHOST=kengjojo-aie494-project.hf.space -JPORT=443 \
  -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" \
  -l results/cloud_results.jtl \
  -e -o results/cloud_dashboard
```

Open the generated dashboard:

```text
results/cloud_dashboard/index.html
```

## Regenerate Dashboard from JTL

The output directory must be empty or non-existent.

```bash
jmeter -g results/local_results.jtl -o results/local_dashboard
```

## Configurable Variables

| Variable | Default | Description |
| --- | --- | --- |
| `PROTOCOL` | `http` | `http` or `https` |
| `HOST` | `localhost` | Target hostname |
| `PORT` | `8000` | Target port |
| `IMAGE_PATH` | none | Absolute path to the image file |

## Test Plan Details

- 50 concurrent users
- 10 second ramp-up
- 10 loops per user
- 500 total requests
- Endpoint: `POST /predict`
- Request body: `multipart/form-data` with the `file` field
- Assertions: HTTP 200 and JSON `predictions` exists

## Metrics to Copy into the Report

After opening the HTML dashboard, copy these values into the report:

- Total requests
- Throughput
- Average response time
- P95 latency
- Error rate

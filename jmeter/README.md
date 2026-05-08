# JMeter Load Testing Guide

## Prerequisites

- [Apache JMeter 5.6+](https://jmeter.apache.org/download_jmeter.cgi) installed
- The API running (locally via Docker or on Hugging Face Spaces)
- An image file locally available on your machine

## A. Start the API with Docker

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

## B. Open JMeter GUI

```bash
jmeter
```

Then open `jmeter/image_classification_load_test.jmx`.

## C. Run the Test Plan

### Option 1: JMeter GUI (for debugging)

1. Open the `.jmx` file in JMeter GUI
2. Click the green **Start** button
3. View results in **View Results Tree** or **Summary Report**

### Option 2: CLI (recommended for actual load testing)

**Local Docker test:**
```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=http -JHOST=localhost -JPORT=8000 \
  -JIMAGE_PATH="C:/path/to/image.jpg" \
  -l results/local_results.jtl \
  -e -o results/local_dashboard
```

**Cloud test (Hugging Face Spaces):**
```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JHOST=<your-hf-space-url-without-https> \
  -JPORT=443 \
  -JPROTOCOL=https \
  -JIMAGE_PATH="C:/path/to/image.jpg" \
  -l results/cloud_results.jtl \
  -e -o results/cloud_dashboard
```

## D. Generate HTML Dashboard

If you already have a `.jtl` file and want to regenerate the dashboard:

```bash
jmeter -g results/local_results.jtl -o results/local_dashboard
```

> **Note:** The output directory must be empty or non-existent.

## E. Configurable Variables

| Variable    | Default         | Description                    |
|-------------|-----------------|--------------------------------|
| `HOST`      | `localhost`     | Target host                    |
| `PORT`      | `8000`          | Target port                    |
| `PROTOCOL`  | `http`          | Protocol (http/https)          |
| `IMAGE_PATH`| *none*            | Path to the test image      |

Override via CLI:
```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JHOST=my-space.hf.space -JPORT=443 -JPROTOCOL=https
```

## F. Test Plan Details

- **Thread Group:** 50 concurrent users
- **Ramp-up:** 10 seconds
- **Loop Count:** 10 (total 500 requests)
- **Endpoint:** `POST /predict` with multipart/form-data
- **Assertions:** HTTP 200 + JSON `$.predictions` exists

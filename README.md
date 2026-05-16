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

โปรเจกต์นี้เป็น API สำหรับจำแนกรูปภาพด้วยโมเดล MobileNetV2 โดยทำเป็น REST API ด้วย FastAPI และ deploy ขึ้น Hugging Face Spaces แบบ Docker

## รายชื่อสมาชิก

1. นายธนนท์ จิตรพรหม 1650904194
2. นายเจตนิพัทธ อินทรีย์ 1650901471
3. นายนิธิกุล แก้วไพฑูรย์ 1650903808

## ลิงก์โปรเจกต์

- GitHub: https://github.com/KengJoJo/AIE494-Project
- Hugging Face Space: https://huggingface.co/spaces/KengJoJo/AIE494-Project
- Cloud API: https://kengjojo-aie494-project.hf.space

## ภาพรวมระบบ

ระบบรับไฟล์รูปภาพจากผู้ใช้ผ่าน endpoint `/predict` แล้วตรวจสอบไฟล์ก่อนนำไปประมวลผล เช่น ตรวจชนิดไฟล์ ขนาดไฟล์ และตรวจว่าเป็นรูปภาพที่เปิดได้จริง จากนั้นส่งรูปเข้าโมเดล MobileNetV2 ที่แปลงเป็น ONNX เพื่อให้ inference บน CPU ได้เร็วขึ้น

โมเดลที่ใช้จริงบน API คือ ONNX ปกติ (`MODEL_TYPE=onnx`) เพราะจากผล benchmark เร็วกว่า PyTorch และให้ผลทำนายที่ใช้งานได้ดีกว่า quantized model ในการทดลองนี้

## โครงสร้างไฟล์

```text
app/                         โค้ด FastAPI
models/                      ไฟล์โมเดล
scripts/                     สคริปต์เตรียมโมเดลและ benchmark
tests/                       unit tests
jmeter/                      JMeter test plan
postman/                     Postman collection
report/                      รายงานโปรเจกต์
results/                     ผล benchmark และผลทดสอบ
.github/workflows/ci-cd.yml  GitHub Actions
Dockerfile                   Docker image สำหรับ deploy
docker-compose.yml           ใช้รัน local ด้วย Docker
```

## วิธีติดตั้งและรัน local

ติดตั้ง dependency:

```bash
pip install -r requirements.txt
```

รัน API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

เปิด Swagger UI:

```text
http://localhost:8000/docs
```

หรือรันด้วย Docker:

```bash
docker compose up --build
```

## Endpoint หลัก

| Method | Endpoint | รายละเอียด |
| --- | --- | --- |
| GET | `/` | ตรวจว่า API ทำงานอยู่ |
| GET | `/health` | ตรวจสถานะโมเดล |
| POST | `/predict` | ส่งรูปภาพเพื่อจำแนกประเภท |

## ตัวอย่าง cURL

Local:

```bash
curl.exe -X POST "http://localhost:8000/predict" -H "accept: application/json" -F "file=@test_cat.jpg;type=image/jpeg"
```

Cloud:

```bash
curl.exe -X POST "https://kengjojo-aie494-project.hf.space/predict" -H "accept: application/json" -F "file=@test_cat.jpg;type=image/jpeg"
```

ตัวอย่างผลลัพธ์:

```json
{
  "filename": "test_cat.jpg",
  "content_type": "image/jpeg",
  "model_type": "onnx",
  "latency_ms": 7712.93,
  "predictions": [
    {
      "label": "lynx, catamount",
      "score": 0.6373
    }
  ]
}
```

## Postman

สามารถ import collection นี้เข้า Postman ได้:

```text
postman/image-classification-api.postman_collection.json
```

ค่า `baseUrl` ที่ใช้ได้:

```text
http://localhost:8000
https://kengjojo-aie494-project.hf.space
```

## การทดสอบ

รัน unit test:

```bash
pytest -q
```

ผลล่าสุด:

```text
17 passed
```

## Benchmark โมเดล

คำสั่งที่ใช้:

```bash
python scripts/benchmark.py --image test_cat.jpg --warmup 20 --runs 50
```

ผลที่ได้:

| Model Type | Size (MB) | Avg (ms) | P50 (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original PyTorch | 13.54 | 17.74 | 17.32 | 22.22 |
| ONNX | 0.33 | 2.23 | 2.19 | 2.60 |
| Quantized ONNX | 3.60 | 20.10 | 19.36 | 22.25 |

ไฟล์ผลลัพธ์อยู่ที่:

```text
results/benchmark_results.md
results/benchmark_results.csv
```

## JMeter Load Test

ไฟล์ test plan:

```text
jmeter/image_classification_load_test.jmx
```

รัน cloud load test:

```bash
jmeter -n -t jmeter/image_classification_load_test.jmx -JPROTOCOL=https -JHOST=kengjojo-aie494-project.hf.space -JPORT=443 -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" -l results/cloud_results.jtl -e -o results/cloud_dashboard
```

ถ้าไม่ได้ตั้ง `jmeter` ไว้ใน PATH ให้ใช้ path ของ `jmeter.bat` แทน เช่น:

```bash
C:/Work/AIE494/apache-jmeter-5.6.3/bin/jmeter.bat -n -t jmeter/image_classification_load_test.jmx -JPROTOCOL=https -JHOST=kengjojo-aie494-project.hf.space -JPORT=443 -JIMAGE_PATH="C:/Work/AIE494/ProjectAIE494/test_cat.jpg" -l results/cloud_results.jtl -e -o results/cloud_dashboard
```

ผล cloud load test ล่าสุด:

| Metric | Value |
| --- | ---: |
| Total Requests | 500 |
| Success | 499 |
| Error | 1 |
| Error Rate | 0.20% |
| Throughput | 5.32 req/s |
| Average Response Time | 7,946.01 ms |
| P95 Latency | 9,768.00 ms |

เปิด dashboard ได้ที่:

```text
results/cloud_dashboard/index.html
```

## CI/CD

ใช้ GitHub Actions ที่ไฟล์:

```text
.github/workflows/ci-cd.yml
```

ขั้นตอนหลักของ pipeline:

1. ติดตั้ง Python และ dependencies
2. รัน `pytest`
3. เตรียมไฟล์โมเดล
4. deploy ไป Hugging Face Spaces เมื่อ push เข้า branch `main`

Secrets ที่ใช้:

| Secret | รายละเอียด |
| --- | --- |
| `HF_TOKEN` | token สำหรับ Hugging Face |
| `HF_SPACE_ID` | repo id ของ Space |

## ไฟล์สำหรับส่งงาน

- รายงาน PDF: `report/AIE494_Project_Report_TH.pdf`
- JMeter test plan: `jmeter/image_classification_load_test.jmx`
- JMeter dashboard: `results/cloud_dashboard/`
- Postman collection: `postman/image-classification-api.postman_collection.json`
- Benchmark results: `results/benchmark_results.md`, `results/benchmark_results.csv`
- GitHub repository: https://github.com/KengJoJo/AIE494-Project
- Cloud API: https://kengjojo-aie494-project.hf.space

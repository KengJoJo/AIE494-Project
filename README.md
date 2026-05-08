---
title: AIE494 Image Classification API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# High-Throughput Image Classification API

โปรเจกต์ API สำหรับจำแนกรูปภาพ (Image Classification) ด้วยโมเดล **MobileNetV2** โดยเน้นความเร็วและประสิทธิภาพสูงด้วย **FastAPI**, **ProcessPoolExecutor** (สำหรับการทำงานแบบ Concurrent) และ **ONNX Runtime** พร้อมกับ **Dynamic Quantization** เพื่อรันบน CPU ได้อย่างรวดเร็ว

---

## 🚀 เริ่มต้นใช้งาน (Quick Start)

### 1. ติดตั้ง Packages
แนะนำให้สร้าง Virtual Environment ก่อน:
```bash
python -m venv .venv
.venv\Scripts\activate   # สำหรับ Windows
pip install -r requirements.txt
```

### 2. เตรียมโมเดล (Model Preparation)
รันสคริปต์ตามลำดับเพื่อดาวน์โหลดและแปลงโมเดลให้อยู่ในรูปแบบ ONNX และ Quantized ONNX:
```bash
# โหลดโมเดล PyTorch จาก Hugging Face
python scripts/download_model.py

# แปลงโมเดลเป็น ONNX
python scripts/export_onnx.py

# บีบอัดโมเดล (Quantization) ให้มีขนาดเล็กลงและเร็วขึ้น
python scripts/quantize_onnx.py
```
*(Option: หากต้องการสร้างรูปภาพทดสอบแบบสุ่ม สามารถรัน `python scripts/generate_sample_image.py` ได้)*

---

## 💻 การรัน API (Running the API)

**รันแบบ Local (สำหรับทดสอบพัฒนา):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
เปิดดูหน้าตา API (Swagger UI) ได้ที่: `http://localhost:8000/docs`

**รันด้วย Docker Compose (สำหรับใช้งานจริง):**
```bash
docker compose up --build -d
```

---

## 📡 วิธีใช้งาน API (API Usage)

ใช้คำสั่ง cURL เพื่อส่งรูปภาพไปทดสอบ (อย่าลืมเปลี่ยน `path/to/image.jpg` เป็นที่อยู่ไฟล์รูปภาพจริงของคุณ):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/image.jpg;type=image/jpeg"
```

**ตัวอย่างผลลัพธ์ (Response):**
```json
{
  "filename": "image.jpg",
  "content_type": "image/jpeg",
  "model_type": "quantized_onnx",
  "latency_ms": 35.42,
  "predictions": [
    {"label": "golden retriever", "score": 0.85},
    {"label": "Labrador retriever", "score": 0.10}
  ]
}
```

---

## 🧪 การทดสอบและวัดประสิทธิภาพ (Testing & Benchmarking)

### 1. Unit Test
ทดสอบการทำงานของ API ว่าถูกต้องหรือไม่:
```bash
pytest -q
```

### 2. Benchmark
เปรียบเทียบความเร็วระหว่างโมเดล Original (PyTorch), ONNX, และ Quantized ONNX:
```bash
python scripts/benchmark.py --image path/to/image.jpg
```
*ผลลัพธ์จะถูกบันทึกลงในโฟลเดอร์ `results/` (เป็นไฟล์ `.csv` และ `.md`)*

### 3. Load Testing (ด้วย JMeter)
ทดสอบจำลองการใช้งานพร้อมกันจำนวนมาก:
```bash
jmeter -n -t jmeter/image_classification_load_test.jmx \
  -JPROTOCOL=http -JHOST=localhost -JPORT=8000 \
  -JIMAGE_PATH="C:/path/to/image.jpg" \
  -l results/local_results.jtl \
  -e -o results/local_dashboard
```

---

## 📦 โครงสร้างโปรเจกต์แบบย่อ (Project Structure)
- `app/`: โค้ดหลักของ FastAPI (Endpoints, Validation, Worker, Inference)
- `models/`: ที่เก็บไฟล์โมเดลที่ดาวน์โหลดมา
- `scripts/`: สคริปต์สำหรับจัดการโมเดลและ Benchmark
- `tests/`: สคริปต์สำหรับ Unit test ด้วย Pytest
- `jmeter/`: ไฟล์ตั้งค่าการเทสต์โหลดแบบจำลองผู้ใช้
- `report/`: ไฟล์เทมเพลตสำหรับเขียนรายงานส่งอาจารย์

"""
API endpoint tests.
Tests GET /, GET /health, and POST /predict.
"""

import os
import sys
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


def _create_test_image_bytes(fmt: str = "JPEG") -> bytes:
    """Generate a small valid test image in memory."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
# GET /
# ──────────────────────────────────────────────
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "running" in data["message"].lower()


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_type" in data


# ──────────────────────────────────────────────
# POST /predict — with mock inference
# ──────────────────────────────────────────────
def _mock_predict_image_bytes(image_bytes, model_type="quantized", model_dir="models", top_k=5):
    """Mock inference function for testing without a real model."""
    return {
        "predictions": [
            {"label": "golden retriever", "score": 0.85},
            {"label": "Labrador retriever", "score": 0.10},
            {"label": "cocker spaniel", "score": 0.03},
        ],
        "model_type": "quantized_onnx",
    }


@patch("app.main.predict_image_bytes", side_effect=_mock_predict_image_bytes)
def test_predict_valid_image(mock_fn):
    """POST /predict with a valid JPEG image should return 200 + predictions."""
    image_bytes = _create_test_image_bytes("JPEG")
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0
    for pred in data["predictions"]:
        assert "label" in pred
        assert "score" in pred
    assert "latency_ms" in data
    assert data["filename"] == "test.jpg"
    assert data["content_type"] == "image/jpeg"


@patch("app.main.predict_image_bytes", side_effect=_mock_predict_image_bytes)
def test_predict_png_image(mock_fn):
    """POST /predict with a valid PNG image should also work."""
    image_bytes = _create_test_image_bytes("PNG")
    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data

"""
Validation tests — ensures proper rejection of invalid uploads.
"""

import os
import sys
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


def _mock_predict(*args, **kwargs):
    return {
        "predictions": [{"label": "test", "score": 0.99}],
        "model_type": "quantized_onnx",
    }


# ──────────────────────────────────────────────
# Reject non-image file (text file)
# ──────────────────────────────────────────────
def test_reject_text_file():
    """Uploading a .txt file should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("document.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"].lower()


# ──────────────────────────────────────────────
# Reject corrupted image
# ──────────────────────────────────────────────
def test_reject_corrupted_image():
    """Uploading random bytes as JPEG should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"not_a_real_image_content", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "not a valid" in response.json()["detail"].lower()


# ──────────────────────────────────────────────
# Reject oversized file
# ──────────────────────────────────────────────
@patch("app.validation.settings")
def test_reject_oversized_file(mock_settings):
    """Uploading a file larger than MAX_UPLOAD_SIZE_MB should return 413."""
    # Set a tiny limit for testing (1 byte)
    mock_settings.MAX_UPLOAD_SIZE_MB = 0  # 0 MB = reject everything

    # Create a small valid image
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    image_bytes = buf.read()

    response = client.post(
        "/predict",
        files={"file": ("large.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


# ──────────────────────────────────────────────
# Reject unsupported content type
# ──────────────────────────────────────────────
def test_reject_pdf():
    """Uploading a PDF should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("document.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    assert response.status_code == 400


# ──────────────────────────────────────────────
# Missing file field → 422
# ──────────────────────────────────────────────
def test_missing_file_field():
    """POST /predict without a file field should return 422."""
    response = client.post("/predict")
    assert response.status_code == 422

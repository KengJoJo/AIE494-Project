"""
Inference utility tests — preprocessing, postprocessing, and label loading.
These tests do NOT require a real model, only testing utility functions.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import load_labels, preprocess_image, postprocess_predictions


# ──────────────────────────────────────────────
# preprocess_image
# ──────────────────────────────────────────────
def test_preprocess_output_shape():
    """Preprocessed image should have shape (1, 3, 224, 224)."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = preprocess_image(image_bytes)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32


def test_preprocess_normalization_range():
    """After ImageNet normalization, values should be roughly in [-3, 3]."""
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = preprocess_image(image_bytes)
    # All-black image normalized: (0/255 - mean) / std
    assert result.min() >= -5.0
    assert result.max() <= 5.0


def test_preprocess_custom_size():
    """Should resize to the specified image_size."""
    img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = preprocess_image(image_bytes, image_size=128)
    assert result.shape == (1, 3, 128, 128)


# ──────────────────────────────────────────────
# postprocess_predictions
# ──────────────────────────────────────────────
def test_postprocess_predictions_structure():
    """Postprocess should return top-K dicts with 'label' and 'score'."""
    logits = np.array([[2.0, 5.0, 1.0, 0.5, 3.0]])
    id2label = {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "horse"}

    results = postprocess_predictions(logits, id2label, top_k=3)
    assert len(results) == 3
    for item in results:
        assert "label" in item
        assert "score" in item
        assert isinstance(item["score"], float)
    # Top prediction should be "dog" (highest logit)
    assert results[0]["label"] == "dog"


def test_postprocess_scores_sum_to_one():
    """Softmax scores across all classes should sum to ~1.0."""
    logits = np.random.randn(1, 10).astype(np.float32)
    id2label = {i: f"class_{i}" for i in range(10)}

    results = postprocess_predictions(logits, id2label, top_k=10)
    total = sum(r["score"] for r in results)
    assert abs(total - 1.0) < 0.01


def test_postprocess_top_k_limit():
    """Should only return top_k results even if there are more classes."""
    logits = np.random.randn(1, 1000).astype(np.float32)
    id2label = {i: f"class_{i}" for i in range(1000)}

    results = postprocess_predictions(logits, id2label, top_k=5)
    assert len(results) == 5


# ──────────────────────────────────────────────
# load_labels
# ──────────────────────────────────────────────
def test_load_labels_from_id2label_json():
    """Should load labels from id2label.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels = {"0": "cat", "1": "dog", "2": "bird"}
        with open(os.path.join(tmpdir, "id2label.json"), "w") as f:
            json.dump(labels, f)

        result = load_labels(tmpdir)
        assert result == {0: "cat", 1: "dog", 2: "bird"}


def test_load_labels_from_config_json():
    """Should fallback to config.json if id2label.json is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"id2label": {"0": "alpha", "1": "beta"}, "model_type": "test"}
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        result = load_labels(tmpdir)
        assert result == {0: "alpha", 1: "beta"}


def test_load_labels_empty_directory():
    """Should return empty dict if no label files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_labels(tmpdir)
        assert result == {}

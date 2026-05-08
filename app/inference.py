"""
Inference utilities — preprocessing and postprocessing for image classification.
Shared by both PyTorch and ONNX inference paths.
"""

import json
import os
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_labels(model_dir: str) -> Dict[int, str]:
    """
    Load id-to-label mapping from a JSON file saved during model download.

    Args:
        model_dir: Path to the model directory containing id2label.json.

    Returns:
        Dictionary mapping class index (int) to human-readable label (str).
    """
    label_path = os.path.join(model_dir, "id2label.json")
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Keys in JSON are strings — convert to int
        return {int(k): v for k, v in raw.items()}

    # Fallback: try config.json (Hugging Face format)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "id2label" in config:
            return {int(k): v for k, v in config["id2label"].items()}

    return {}


from transformers import AutoImageProcessor

# Global singleton for the processor (cached per worker process)
_PROCESSOR = None


def get_processor(model_dir: str = None) -> AutoImageProcessor:
    """
    Get the AutoImageProcessor instance, loading it once if necessary.

    Args:
        model_dir: Path to the model directory. If None, resolves to ../models/original.

    Returns:
        AutoImageProcessor instance.
    """
    global _PROCESSOR
    if _PROCESSOR is None:
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "original")
        _PROCESSOR = AutoImageProcessor.from_pretrained(model_dir)
    return _PROCESSOR


def preprocess_image(image_bytes: bytes, model_dir: str = None) -> np.ndarray:
    """
    Preprocess raw image bytes using the official Hugging Face ImageProcessor.
    Ensures resizing, cropping, and normalization match the model's training requirements.

    Args:
        image_bytes: Raw image file bytes.
        model_dir: Optional path to the model directory for loading the processor config.

    Returns:
        np.ndarray of shape (1, 3, 224, 224), dtype float32.
    """
    processor = get_processor(model_dir)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # The processor handles resizing (e.g. 256 then 224 center crop) and normalization
    # Note: Fast processors may only support return_tensors="pt"
    inputs = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].numpy().astype(np.float32)


def postprocess_predictions(
    logits: np.ndarray,
    id2label: Dict[int, str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """
    Convert raw model logits into top-K classification results.

    Args:
        logits: Raw output array of shape (1, num_classes).
        id2label: Mapping from class index to label string.
        top_k: Number of top predictions to return.

    Returns:
        List of dicts with "label" and "score" keys, sorted by score desc.
    """
    # Softmax
    logits = logits[0]  # Remove batch dimension
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # Top-K indices
    top_indices = probs.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        label = id2label.get(int(idx), f"class_{idx}")
        score = float(probs[idx])
        results.append({"label": label, "score": round(score, 4)})

    return results

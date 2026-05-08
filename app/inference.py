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


def preprocess_image(image_bytes: bytes, image_size: int = 224) -> np.ndarray:
    """
    Preprocess raw image bytes into a normalized float32 numpy array
    suitable for MobileNetV2 / similar ImageNet models.

    Steps:
        1. Open & convert to RGB
        2. Resize to (image_size, image_size)
        3. Convert to float32 and scale to [0, 1]
        4. Normalize with ImageNet mean and std
        5. Transpose to NCHW format (1, 3, H, W)

    Args:
        image_bytes: Raw image file bytes.
        image_size: Target spatial dimension (default 224 for MobileNetV2).

    Returns:
        np.ndarray of shape (1, 3, image_size, image_size), dtype float32.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # To numpy float32 [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr


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

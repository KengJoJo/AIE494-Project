"""
Worker module for CPU-bound inference in a separate process.

This module is designed to be used with ProcessPoolExecutor.
The ONNX Runtime session and labels are loaded ONCE per worker process
(via module-level globals + lazy init), avoiding expensive re-loading
on every request.
"""

import os
from typing import Any, Dict, List, Optional

import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)

# Global per-process state — initialized lazily on first call
_ort_session = None
_id2label: Dict[int, str] = {}
_model_type: str = "onnx"
_top_k: int = 5

# Default model paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "onnx", "model.onnx")
DEFAULT_MODEL_TYPE = "onnx"


def _get_model_path(model_type: str, model_dir: str = "models") -> str:
    """Resolve the model file path based on model type."""
    # Production uses standard ONNX because it had the best benchmark result.
    return os.path.join(model_dir, "onnx", "model.onnx")


def _init_worker(model_type: str = "onnx", model_dir: str = "models", top_k: int = 5):
    """
    Lazily initialize the ONNX Runtime session and label map.
    Called once per worker process on the first inference request.
    """
    global _ort_session, _id2label, _model_type, _top_k
    import onnxruntime as ort
    from app.inference import load_labels

    _model_type = model_type
    _top_k = top_k

    model_path = _get_model_path(model_type, model_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run the model preparation scripts first."
        )

    print(f"INFO:  Worker process initializing with model: {model_path}")
    sys.stdout.flush()

    # Create ONNX Runtime session with CPU provider
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Load labels from the original model directory (always available)
    _id2label = load_labels(os.path.join(model_dir, "original"))
    
    print(f"INFO:  Worker ready. Labels: {len(_id2label)}")
    sys.stdout.flush()


def predict_image_bytes(
    image_bytes: bytes,
    model_type: str = "onnx",
    model_dir: str = "models",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Run inference on raw image bytes. This function is called inside a
    worker process via ProcessPoolExecutor.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        model_type: Which model variant to use.
        model_dir: Root directory for model artifacts.
        top_k: Number of top predictions to return.

    Returns:
        Dict with "predictions" (list) and "model_type" (str).
    """
    global _ort_session, _id2label

    # Lazy init on first call in this process
    if _ort_session is None:
        _init_worker(model_type, model_dir, top_k)

    from app.inference import preprocess_image, postprocess_predictions

    # Preprocess (passing the original model dir to load the correct processor config)
    input_array = preprocess_image(image_bytes, os.path.join(model_dir, "original"))

    # Run ONNX inference
    input_name = _ort_session.get_inputs()[0].name
    outputs = _ort_session.run(None, {input_name: input_array})
    logits = outputs[0]

    # Postprocess
    predictions = postprocess_predictions(logits, _id2label, top_k)

    return {
        "predictions": predictions,
        "model_type": model_type,
    }

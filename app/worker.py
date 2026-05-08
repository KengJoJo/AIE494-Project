"""
Worker module for CPU-bound inference in a separate process.

This module is designed to be used with ProcessPoolExecutor.
The ONNX Runtime session and labels are loaded ONCE per worker process
(via module-level globals + lazy init), avoiding expensive re-loading
on every request.
"""

import os
from typing import Dict, List, Optional

import numpy as np

# Global per-process state — initialized lazily on first call
_ort_session = None
_id2label: Dict[int, str] = {}
_model_type: str = "quantized"
_top_k: int = 5


def _get_model_path(model_type: str, model_dir: str = "models") -> str:
    """Resolve the model file path based on model type."""
    if model_type == "quantized":
        return os.path.join(model_dir, "quantized", "model_quantized.onnx")
    elif model_type == "onnx":
        return os.path.join(model_dir, "onnx", "model.onnx")
    else:
        # For 'original' type, we still use ONNX for worker-based inference
        # The original PyTorch model is only used in benchmark scripts
        return os.path.join(model_dir, "onnx", "model.onnx")


def _init_worker(model_type: str = "quantized", model_dir: str = "models", top_k: int = 5):
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

    # Create ONNX Runtime session with CPU provider
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 2
    _ort_session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Load labels from the original model directory (always available)
    _id2label = load_labels(os.path.join(model_dir, "original"))
    if not _id2label:
        # Try loading from the onnx directory as fallback
        _id2label = load_labels(os.path.join(model_dir, "onnx"))


def predict_image_bytes(
    image_bytes: bytes,
    model_type: str = "quantized",
    model_dir: str = "models",
    top_k: int = 5,
) -> Dict:
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

    # Preprocess
    input_array = preprocess_image(image_bytes)

    # Run ONNX inference
    input_name = _ort_session.get_inputs()[0].name
    outputs = _ort_session.run(None, {input_name: input_array})
    logits = outputs[0]

    # Postprocess
    predictions = postprocess_predictions(logits, _id2label, top_k)

    actual_type = model_type
    if model_type == "quantized":
        actual_type = "quantized_onnx"
    elif model_type == "onnx":
        actual_type = "onnx"

    return {
        "predictions": predictions,
        "model_type": actual_type,
    }

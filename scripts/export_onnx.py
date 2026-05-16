"""
Export the PyTorch MobileNetV2 model to ONNX format.

Usage:
    python scripts/export_onnx.py
"""

import os
import sys

import numpy as np
import torch


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from transformers import AutoModelForImageClassification

    original_dir = os.path.join(project_root, "models", "original")
    onnx_dir = os.path.join(project_root, "models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    if not os.path.isdir(original_dir):
        print(f"ERROR: Original model not found at {original_dir}")
        print("Run 'python scripts/download_model.py' first.")
        sys.exit(1)

    print(f"Loading PyTorch model from: {original_dir}")
    model = AutoModelForImageClassification.from_pretrained(original_dir)
    model.eval()

    # Create dummy input (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting ONNX model to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
    )

    # Copy id2label.json to onnx dir for convenience
    import shutil
    id2label_src = os.path.join(original_dir, "id2label.json")
    if os.path.exists(id2label_src):
        shutil.copy2(id2label_src, os.path.join(onnx_dir, "id2label.json"))

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model verification passed.")

    # Quick inference test
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_np = dummy_input.numpy()
    outputs = session.run(None, {input_name: dummy_np})
    print(f"[OK] ONNX inference test passed. Output shape: {outputs[0].shape}")

    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX model size: {model_size:.2f} MB")
    print("Done! ONNX model saved to:", onnx_path)


if __name__ == "__main__":
    main()

"""
Download the MobileNetV2 image classification model from Hugging Face
and save all artifacts to models/original/.

Usage:
    python scripts/download_model.py
"""

import json
import os
import sys

def main():
    # Ensure project root is on sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_name = "google/mobilenet_v2_1.0_224"
    save_dir = os.path.join(project_root, "models", "original")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Saving to: {save_dir}")

    # Download and save the image processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.save_pretrained(save_dir)
    print("[OK] Image processor saved.")

    # Download and save the model
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print("[OK] Model saved.")

    # Save id2label mapping for inference
    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = model.config.id2label
        label_path = os.path.join(save_dir, "id2label.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(id2label, f, indent=2)
        print(f"[OK] Labels saved ({len(id2label)} classes).")
    else:
        print("[WARN] No id2label found in model config.")

    print("\nDone! Model artifacts saved to:", save_dir)


if __name__ == "__main__":
    main()

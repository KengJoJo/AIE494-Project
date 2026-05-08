"""
Generate a sample test image for the project.
Run this if you don't have sample_image.jpg yet.

Usage:
    python scripts/generate_sample_image.py
"""

import os
from PIL import Image, ImageDraw


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "sample_image.jpg")

    # Create a colorful test image (224x224)
    img = Image.new("RGB", (224, 224), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Draw some shapes to make it more interesting
    draw.rectangle([20, 120, 204, 204], fill=(34, 139, 34))   # Green ground
    draw.ellipse([70, 30, 154, 114], fill=(255, 223, 0))       # Yellow sun
    draw.rectangle([90, 140, 130, 200], fill=(139, 69, 19))    # Brown tree trunk
    draw.ellipse([60, 100, 160, 160], fill=(0, 100, 0))        # Dark green tree top

    img.save(output_path, "JPEG", quality=95)
    print(f"[OK] Sample image saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path)} bytes")


if __name__ == "__main__":
    main()

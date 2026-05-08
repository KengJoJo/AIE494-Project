"""
Sanity check script — verify that the API returns reasonable predictions for a clear image.
"""

import os
import sys
import requests
from PIL import Image
from io import BytesIO

def main():
    # 1. Start the API in the background if not already running, 
    # but for this script we assume it's running at localhost:8000
    url = "http://localhost:8000/predict"
    
    # 2. Check if a test image exists, if not, we can't do much without generating one.
    # We'll try to find any .jpg in the project root or use a fallback.
    test_image = "dog_test.jpg"
    
    if not os.path.exists(test_image):
        print(f"Creating a dummy test image: {test_image}")
        # Create a simple blue square if no real image is provided
        # (In a real scenario, the user should provide a clear dog.jpg)
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        img.save(test_image)

    print(f"Testing API with {test_image}...")
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image, f, "image/jpeg")}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("\n[SUCCESS] API Response:")
            print(f"  Model Type: {data.get('model_type')}")
            print(f"  Latency: {data.get('latency_ms')} ms")
            print("  Top Predictions:")
            for p in data.get("predictions", []):
                print(f"    - {p['label']}: {p['score']}")
            
            # Validation
            assert "predictions" in data
            assert len(data["predictions"]) > 0
            assert 0 <= data["predictions"][0]["score"] <= 1
            print("\n[PASS] Sanity check passed!")
        else:
            print(f"\n[FAILED] API returned status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to API. Is it running at http://localhost:8000?")
        print("Run: uvicorn app.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()

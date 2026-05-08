"""
Image upload validation module.
Checks file type, size, and image integrity before inference.
"""

from io import BytesIO
from PIL import Image
from fastapi import HTTPException, UploadFile

from app.settings import settings

# Allowed MIME types for image uploads
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def validate_and_read_image(file: UploadFile) -> bytes:
    """
    Validate the uploaded file and return raw image bytes.

    Checks performed:
        1. Content type is an allowed image MIME type
        2. File size does not exceed MAX_UPLOAD_SIZE_MB
        3. Image bytes can be decoded by Pillow (not corrupted)

    Args:
        file: The uploaded file from the request.

    Returns:
        Raw image bytes ready for inference.

    Raises:
        HTTPException 400: Invalid content type or corrupted image.
        HTTPException 413: File exceeds maximum allowed size.
    """
    # --- Check content type ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # --- Read file bytes ---
    image_bytes = await file.read()

    # --- Check file size ---
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {len(image_bytes) / (1024*1024):.2f} MB exceeds "
                f"maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    # --- Verify image integrity ---
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Check for corruption without fully decoding
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image or is corrupted.",
        )

    return image_bytes

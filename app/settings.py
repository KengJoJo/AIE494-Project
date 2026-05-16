"""
Application settings managed via environment variables.
Uses pydantic-settings for type-safe configuration.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration loaded from environment variables with sensible defaults.

    Environment variables:
        MODEL_TYPE      : "onnx" | "quantized" (default: onnx)
        MODEL_DIR       : Root directory for model artifacts (default: models)
        MAX_UPLOAD_SIZE_MB : Maximum upload file size in megabytes (default: 5)
        TOP_K           : Number of top predictions to return (default: 5)
        WORKERS         : Number of worker processes for inference (default: 2)
    """

    MODEL_TYPE: str = "onnx"
    MODEL_DIR: str = "models"
    MAX_UPLOAD_SIZE_MB: int = 5
    TOP_K: int = 5
    WORKERS: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance used throughout the app
settings = Settings()

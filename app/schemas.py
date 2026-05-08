"""
Pydantic response schemas for the Image Classification API.
Ensures consistent, typed JSON responses across all endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """A single classification prediction."""
    label: str = Field(..., description="Predicted class label")
    score: float = Field(..., description="Confidence score (0-1)")


class PredictResponse(BaseModel):
    """Response for POST /predict."""
    filename: str = Field(..., description="Original uploaded filename")
    content_type: str = Field(..., description="MIME type of the uploaded file")
    model_type: str = Field(..., description="Model variant used (original/onnx/quantized)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    predictions: List[PredictionItem] = Field(..., description="Top-K predictions")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_type: str = Field(..., description="Active model type")


class RootResponse(BaseModel):
    """Response for GET /."""
    status: str = Field(default="ok")
    message: str = Field(
        default="High-Throughput Image Classification API is running"
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str = Field(..., description="Error message")

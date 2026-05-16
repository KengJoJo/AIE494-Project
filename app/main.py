"""
FastAPI application - High-Throughput Image Classification API.

Endpoints:
    GET  /         -> Service status
    GET  /health   -> Health check with model status
    POST /predict  -> Image classification inference
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictResponse,
    RootResponse,
)
from app.settings import settings
from app.validation import validate_and_read_image
from app.worker import predict_image_bytes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process pool for CPU-bound inference
# ---------------------------------------------------------------------------
executor: Optional[ProcessPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage ProcessPoolExecutor lifecycle."""
    global executor
    executor = ProcessPoolExecutor(max_workers=settings.WORKERS)
    logger.info(
        "Started ProcessPoolExecutor with %d workers | model_type=%s",
        settings.WORKERS,
        settings.MODEL_TYPE,
    )
    yield
    executor.shutdown(wait=True)
    logger.info("ProcessPoolExecutor shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="High-Throughput Image Classification API",
    description=(
        "A production-ready image classification service powered by "
        "MobileNetV2 with ONNX Runtime for fast CPU inference."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["General"])
async def root():
    """Service status endpoint."""
    return RootResponse()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check - reports whether the model is loaded."""
    import os
    from app.worker import _get_model_path

    model_path = _get_model_path(settings.MODEL_TYPE, settings.MODEL_DIR)
    model_exists = os.path.exists(model_path)

    model_type_label = settings.MODEL_TYPE
    if settings.MODEL_TYPE == "quantized":
        model_type_label = "quantized_onnx"

    return HealthResponse(
        status="healthy" if model_exists else "unhealthy",
        model_loaded=model_exists,
        model_type=model_type_label,
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    tags=["Inference"],
)
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image.

    - Validates the file (type, size, integrity)
    - Dispatches CPU-bound inference to a worker process
    - Returns top-K predictions with latency measurement
    """
    # 1. Validate and read image bytes
    image_bytes = await validate_and_read_image(file)

    # 2. Dispatch inference to worker process
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    try:
        result = await loop.run_in_executor(
            executor,
            predict_image_bytes,
            image_bytes,
            settings.MODEL_TYPE,
            settings.MODEL_DIR,
            settings.TOP_K,
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Inference error: {str(e)}"},
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # 3. Build response
    return PredictResponse(
        filename=file.filename or "unknown",
        content_type=file.content_type or "unknown",
        model_type=result["model_type"],
        latency_ms=round(elapsed_ms, 2),
        predictions=result["predictions"],
    )

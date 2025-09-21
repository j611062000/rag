from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.api.routes import router
from app.startup import run_startup_ingestion


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class IngestRequest(BaseModel):
    file_path: str
    session_id: Optional[str] = "default"


class ClearRequest(BaseModel):
    session_id: Optional[str] = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Chat-with-PDF Backend starting up...")

    # Run startup ingestion
    try:
        await run_startup_ingestion()
    except Exception as e:
        logger.error(f"Startup ingestion failed: {str(e)}")

    logger.info("✅ Application ready to serve requests")
    yield

    # Shutdown
    logger.info("📄 Chat-with-PDF Backend shutting down...")


app = FastAPI(
    title="Chat with PDF Backend",
    description="Intelligent Q&A over academic PDFs with multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Initialize Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
from loguru import logger

from app.graph.orchestrator import ChatOrchestrator
from app.rag.ingestor import PDFIngestor
from app.memory.session import SessionManager

router = APIRouter()

orchestrator = ChatOrchestrator()
ingestor = PDFIngestor()
session_manager = SessionManager()


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class IngestRequest(BaseModel):
    file_path: str
    session_id: Optional[str] = "default"


class ClearRequest(BaseModel):
    session_id: Optional[str] = "default"


@router.post("/ask")
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    try:
        logger.info(f"Processing question: {request.question[:100]}...")

        result = await orchestrator.process_query(
            question=request.question,
            session_id=request.session_id
        )

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()
        result = await ingestor.ingest_pdf_content(content, file.filename)

        return {
            "message": f"Successfully ingested {file.filename}",
            "document_id": result.get("document_id"),
            "chunks_created": result.get("chunks_created", 0)
        }
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_session(request: ClearRequest) -> Dict[str, str]:
    try:
        await session_manager.clear_session(request.session_id)
        return {"message": f"Session {request.session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    try:
        history = await session_manager.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
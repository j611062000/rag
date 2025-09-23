from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
from loguru import logger

from app.graph.orchestrator import ChatOrchestrator
from app.rag.ingestor import PDFIngestor
from app.memory.session import SessionManager
from app.rag.vector_store import get_vector_store

router = APIRouter()

orchestrator = ChatOrchestrator()
ingestor = PDFIngestor()
session_manager = SessionManager()
vector_store = get_vector_store()


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
            "session_id": request.session_id,
            "route_used": result.get("route_used", "unknown"),
            "source_attribution": {
                "used_pdf": result.get("used_pdf", False),
                "used_web": result.get("used_web", False),
                "pdf_confidence": result.get("pdf_confidence", 0.0),
                "web_confidence": result.get("web_confidence", 0.0)
            }
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


@router.get("/documents")
async def list_documents(limit: Optional[int] = 100) -> Dict[str, Any]:
    try:
        documents = await vector_store.list_documents(limit=limit)
        return {
            "documents": documents,
            "total_count": len(documents),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error listing documents from vector database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
import hashlib
from typing import Dict, Any, List
from io import BytesIO
import uuid

from pypdf import PdfReader
from loguru import logger

from app.rag.chunker import DocumentChunker
from app.rag.vector_store import get_vector_store


class PDFIngestor:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.vector_store = get_vector_store()

    async def ingest_pdf_content(self, content: bytes, filename: str) -> Dict[str, Any]:
        try:
            # Extract text from PDF
            pdf_reader = PdfReader(BytesIO(content))
            text_content = ""

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

            # Generate document ID
            document_id = hashlib.md5(content).hexdigest()

            # Create metadata
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "total_pages": len(pdf_reader.pages),
                "content_length": len(text_content)
            }

            # Chunk the document
            chunks = self.chunker.chunk_document(text_content, metadata)

            # Prepare for vector store
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            # Add to vector store
            await self.vector_store.add_documents(texts, metadatas, ids)

            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")

            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "total_pages": len(pdf_reader.pages)
            }

        except Exception as e:
            logger.error(f"Error ingesting PDF {filename}: {str(e)}")
            raise

    async def ingest_pdf_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            content = f.read()

        filename = file_path.split("/")[-1]
        return await self.ingest_pdf_content(content, filename)
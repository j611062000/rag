import hashlib
from typing import Dict, Any
from io import BytesIO

from pypdf import PdfReader
from loguru import logger

from app.rag.chunker import DocumentChunker
from app.rag.vector_store import get_vector_store
from app.rag.advanced_parser import AdvancedPDFParser
from app.rag.semantic_chunker import get_semantic_chunker


class PDFIngestor:
    def __init__(self, use_advanced_parsing: bool = True):
        self.use_advanced_parsing = use_advanced_parsing
        self.chunker = DocumentChunker()
        self.semantic_chunker = get_semantic_chunker()
        self.vector_store = get_vector_store()
        self.advanced_parser = AdvancedPDFParser()

    async def ingest_pdf_content(self, content: bytes, filename: str) -> Dict[str, Any]:
        try:
            if self.use_advanced_parsing:
                # Use advanced parsing with Unstructured.io and semantic chunking
                logger.info(f"Using advanced parsing for {filename}")
                result = await self.advanced_parser.ingest_pdf_content(content, filename)
                logger.info(f"Advanced parsing completed for {filename}")
                return result
            else:
                # Use legacy parsing
                return await self._legacy_ingest(content, filename)
        except Exception as e:
            logger.error(f"Advanced parsing failed for {filename}: {str(e)}, falling back to legacy")
            return await self._legacy_ingest(content, filename)

    async def _legacy_ingest(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback to legacy ingestion method"""
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
                "content_length": len(text_content),
                "parsing_method": "legacy_pypdf"
            }

            # Use semantic chunking if available, otherwise fall back to regular chunking
            try:
                semantic_chunks = self.semantic_chunker.chunk_document(text_content, metadata)
                chunks = [
                    type('Chunk', (), {
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'chunk_id': chunk.chunk_id
                    })()
                    for chunk in semantic_chunks
                ]
                logger.info(f"Used semantic chunking for {filename}")
            except Exception as semantic_e:
                logger.warning(f"Semantic chunking failed: {str(semantic_e)}, using regular chunking")
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
                "total_pages": len(pdf_reader.pages),
                "parsing_method": "legacy_with_semantic_chunking"
            }

        except Exception as e:
            import traceback
            logger.error(f"Error ingesting PDF {filename}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Try to identify which step failed
            try:
                # Test PDF reading
                pdf_reader = PdfReader(BytesIO(content))
                logger.info(f"PDF reading successful: {len(pdf_reader.pages)} pages")

                # Test text extraction
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

                logger.info(f"Text extraction successful: {len(text_content)} chars")

                # Test chunking
                document_id = hashlib.md5(content).hexdigest()
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "total_pages": len(pdf_reader.pages),
                    "content_length": len(text_content)
                }
                chunks = self.chunker.chunk_document(text_content, metadata)
                logger.info(f"Chunking successful: {len(chunks)} chunks created")

                # The error must be in vector store operations
                logger.error("Error likely in vector store add_documents operation")

            except Exception as debug_e:
                logger.error(f"Debug error in step identification: {str(debug_e)}")

            raise

    async def ingest_pdf_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            content = f.read()

        filename = file_path.split("/")[-1]
        return await self.ingest_pdf_content(content, filename)
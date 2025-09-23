"""
Advanced document parser using Unstructured.io for better content extraction
"""
import hashlib
from typing import Dict, Any, List
from io import BytesIO
from dataclasses import dataclass

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
    from unstructured.staging.base import dict_to_elements
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

from pypdf import PdfReader
from loguru import logger

from app.rag.chunker import DocumentChunker
from app.rag.vector_store import get_vector_store


@dataclass
class EnhancedChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    element_type: str  # table, text, title, etc.
    parent_section: str = None


class AdvancedPDFParser:
    def __init__(self):
        self.fallback_chunker = DocumentChunker()
        self.vector_store = get_vector_store()

    async def ingest_pdf_content(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Enhanced PDF parsing with table and structure preservation"""

        if not UNSTRUCTURED_AVAILABLE:
            logger.warning("Unstructured.io not available, falling back to basic parsing")
            return await self._fallback_parsing(content, filename)

        try:
            # Use Unstructured.io for advanced parsing
            elements = partition_pdf(
                file=BytesIO(content),
                strategy="hi_res",  # High resolution for better table detection
                infer_table_structure=True,  # Extract table structure
                extract_images_in_pdf=False,  # Skip images for now
                include_page_breaks=True,
                chunking_strategy="by_title",  # Group by sections
                max_characters=800,  # Match our chunk size
                overlap=240,  # Match our overlap
            )

            # Process elements into enhanced chunks
            enhanced_chunks = self._process_elements(elements, filename)

            # Generate document metadata
            document_id = hashlib.md5(content).hexdigest()
            document_metadata = {
                "document_id": document_id,
                "filename": filename,
                "total_elements": len(elements),
                "content_length": sum(len(chunk.content) for chunk in enhanced_chunks),
                "parsing_method": "unstructured_advanced"
            }

            # Prepare for vector store
            texts = [chunk.content for chunk in enhanced_chunks]
            metadatas = [
                {
                    **document_metadata,
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(enhanced_chunks),
                    "element_type": chunk.element_type,
                    "parent_section": chunk.parent_section
                }
                for i, chunk in enumerate(enhanced_chunks)
            ]
            ids = [chunk.chunk_id for chunk in enhanced_chunks]

            # Add to vector store
            await self.vector_store.add_documents(texts, metadatas, ids)

            logger.info(f"Successfully ingested {filename} with {len(enhanced_chunks)} enhanced chunks")

            return {
                "document_id": document_id,
                "chunks_created": len(enhanced_chunks),
                "parsing_method": "unstructured_advanced",
                "element_types": self._get_element_type_counts(enhanced_chunks)
            }

        except Exception as e:
            logger.error(f"Advanced parsing failed for {filename}: {str(e)}, falling back")
            return await self._fallback_parsing(content, filename)

    def _process_elements(self, elements, filename: str) -> List[EnhancedChunk]:
        """Convert Unstructured elements to enhanced chunks"""
        chunks = []
        current_section = "Introduction"  # Default section

        for i, element in enumerate(elements):
            element_type = element.category.lower() if hasattr(element, 'category') else 'text'

            # Track current section for context
            if element_type == 'title':
                current_section = str(element)[:100]  # Use title as section name

            # Handle tables specially
            if element_type == 'table':
                content = self._format_table_content(element)
            else:
                content = str(element)

            # Skip very short content
            if len(content.strip()) < 10:
                continue

            # Create enhanced chunk
            chunk_metadata = {
                "element_id": getattr(element, 'element_id', f"elem_{i}"),
                "page_number": getattr(element, 'metadata', {}).get('page_number', 1),
                "coordinates": getattr(element, 'metadata', {}).get('coordinates', None),
                "element_type": element_type,
                "parent_section": current_section
            }

            chunk_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{i}"

            chunk = EnhancedChunk(
                content=content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                element_type=element_type,
                parent_section=current_section
            )

            chunks.append(chunk)

        return chunks

    def _format_table_content(self, table_element) -> str:
        """Format table content for better search"""
        try:
            # If table has structured data, format it nicely
            if hasattr(table_element, 'metadata') and 'text_as_html' in table_element.metadata:
                html_content = table_element.metadata['text_as_html']
                # Convert HTML table to readable text
                content = f"TABLE: {str(table_element)}\n\nStructured data: {html_content}"
            else:
                content = f"TABLE: {str(table_element)}"

            return content
        except:
            return f"TABLE: {str(table_element)}"

    def _get_element_type_counts(self, chunks: List[EnhancedChunk]) -> Dict[str, int]:
        """Get count of different element types for reporting"""
        counts = {}
        for chunk in chunks:
            element_type = chunk.element_type
            counts[element_type] = counts.get(element_type, 0) + 1
        return counts

    async def _fallback_parsing(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback to basic PyPDF parsing if Unstructured fails"""
        logger.info(f"Using fallback parsing for {filename}")

        try:
            # Basic PDF parsing
            pdf_reader = PdfReader(BytesIO(content))
            text_content = ""

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

            # Generate document ID and metadata
            document_id = hashlib.md5(content).hexdigest()
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "total_pages": len(pdf_reader.pages),
                "content_length": len(text_content),
                "parsing_method": "pypdf_fallback"
            }

            # Use regular chunking
            chunks = self.fallback_chunker.chunk_document(text_content, metadata)

            # Prepare for vector store
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            # Add to vector store
            await self.vector_store.add_documents(texts, metadatas, ids)

            logger.info(f"Successfully ingested {filename} with {len(chunks)} basic chunks")

            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "parsing_method": "pypdf_fallback",
                "total_pages": len(pdf_reader.pages)
            }

        except Exception as e:
            logger.error(f"Fallback parsing also failed for {filename}: {str(e)}")
            raise

    async def ingest_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest PDF from file path"""
        with open(file_path, "rb") as f:
            content = f.read()

        filename = file_path.split("/")[-1]
        return await self.ingest_pdf_content(content, filename)
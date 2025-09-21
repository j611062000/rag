from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass

from app.config import settings


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = self.splitter.split_text(text)

        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            chunk_id = f"{metadata.get('document_id', 'unknown')}_{i}"

            document_chunks.append(DocumentChunk(
                content=chunk,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        return document_chunks
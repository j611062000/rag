"""
Semantic chunking that preserves context and meaning
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import numpy as np

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from sentence_transformers import SentenceTransformer
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

from app.config import settings
from app.rag.embeddings import get_embedding_provider
from loguru import logger


@dataclass
class SemanticChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    semantic_score: float = 0.0
    boundary_type: str = "semantic"  # semantic, paragraph, sentence


class AdvancedSemanticChunker:
    """Advanced chunking that preserves semantic boundaries and context"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.embedding_provider = get_embedding_provider()

        # Fallback to recursive chunking if semantic chunking not available
        if not SEMANTIC_CHUNKING_AVAILABLE:
            logger.warning("Semantic chunking dependencies not available, using fallback")
            self.use_semantic = False
            self.fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
            )
        else:
            self.use_semantic = True
            # Initialize semantic chunker with embedding model
            self.semantic_splitter = SemanticChunker(
                embeddings=self._get_langchain_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80  # More conservative splitting
            )

    def _get_langchain_embeddings(self):
        """Convert our embedding provider to LangChain format"""
        class EmbeddingWrapper:
            def __init__(self, embedding_provider):
                self.embedding_provider = embedding_provider

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embedding_provider.embed_documents(texts)

            def embed_query(self, text: str) -> List[float]:
                return self.embedding_provider.embed_text(text)

        return EmbeddingWrapper(self.embedding_provider)

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Chunk document using semantic boundaries"""

        if self.use_semantic:
            return self._semantic_chunk(text, metadata)
        else:
            return self._fallback_chunk(text, metadata)

    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Use semantic chunking to preserve meaning"""
        try:
            # Preprocess text to improve chunking
            processed_text = self._preprocess_text(text)

            # Use semantic chunker
            raw_chunks = self.semantic_splitter.split_text(processed_text)

            # Post-process chunks
            semantic_chunks = []
            for i, chunk_text in enumerate(raw_chunks):
                # Calculate semantic coherence score
                semantic_score = self._calculate_semantic_score(chunk_text)

                # Determine boundary type
                boundary_type = self._determine_boundary_type(chunk_text)

                # Create enhanced metadata
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "semantic_score": semantic_score,
                    "boundary_type": boundary_type,
                    "chunk_size_chars": len(chunk_text),
                    "chunking_method": "semantic"
                }

                chunk_id = f"{metadata.get('document_id', 'unknown')}_{i}"

                chunk = SemanticChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    semantic_score=semantic_score,
                    boundary_type=boundary_type
                )

                semantic_chunks.append(chunk)

            # Apply overlap if needed
            if self.chunk_overlap > 0:
                semantic_chunks = self._add_semantic_overlap(semantic_chunks)

            logger.info(f"Created {len(semantic_chunks)} semantic chunks")
            return semantic_chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}, using fallback")
            return self._fallback_chunk(text, metadata)

    def _fallback_chunk(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Fallback to recursive chunking"""
        raw_chunks = self.fallback_splitter.split_text(text)

        semantic_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "semantic_score": 0.5,  # Default score
                "boundary_type": "recursive",
                "chunk_size_chars": len(chunk_text),
                "chunking_method": "recursive_fallback"
            }

            chunk_id = f"{metadata.get('document_id', 'unknown')}_{i}"

            chunk = SemanticChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                semantic_score=0.5,
                boundary_type="recursive"
            )

            semantic_chunks.append(chunk)

        return semantic_chunks

    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for better semantic chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([.!?])\s*\n([A-Z])', r'\1 \2', text)  # Fix sentence breaks

        # Enhance section boundaries
        text = re.sub(r'\n([A-Z][A-Za-z\s]+:)\n', r'\n\n\1\n\n', text)  # Section headers
        text = re.sub(r'\n(Table \d+|Figure \d+)', r'\n\n\1', text)  # Table/Figure markers

        return text.strip()

    def _calculate_semantic_score(self, chunk_text: str) -> float:
        """Calculate semantic coherence score for a chunk"""
        try:
            # Simple heuristics for semantic coherence
            score = 0.5  # Base score

            # Penalty for fragments
            if len(chunk_text.strip()) < 50:
                score -= 0.2

            # Bonus for complete sentences
            sentences = re.split(r'[.!?]+', chunk_text)
            complete_sentences = [s for s in sentences if len(s.strip()) > 10]
            if len(complete_sentences) >= 2:
                score += 0.2

            # Bonus for structured content
            if re.search(r'^\d+\.|\n\d+\.|\n-|\n\*', chunk_text, re.MULTILINE):
                score += 0.1

            # Bonus for topic coherence (repeated key terms)
            words = re.findall(r'\b\w{4,}\b', chunk_text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            if repeated_words > len(words) * 0.1:  # 10% repeated words
                score += 0.1

            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except:
            return 0.5  # Default if calculation fails

    def _determine_boundary_type(self, chunk_text: str) -> str:
        """Determine what type of boundary this chunk represents"""
        text_lower = chunk_text.lower()

        # Check for specific boundary types
        if re.match(r'^\d+\.?\s+(introduction|conclusion|method|result)', text_lower):
            return "section_header"
        elif 'table' in text_lower[:50] and any(char in chunk_text for char in ['|', '\t']):
            return "table"
        elif re.search(r'figure \d+|fig\. \d+', text_lower[:100]):
            return "figure_caption"
        elif re.search(r'\n\n.*:\n', chunk_text):
            return "definition_list"
        elif len(re.findall(r'[.!?]', chunk_text)) >= 3:
            return "paragraph"
        else:
            return "semantic"

    def _add_semantic_overlap(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Add semantic overlap between chunks"""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            content = chunk.content

            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_sentences = re.split(r'[.!?]+', prev_chunk.content)
                if len(prev_sentences) > 1:
                    # Take last sentence from previous chunk
                    overlap_text = prev_sentences[-2] + '. ' if prev_sentences[-2].strip() else ''
                    content = overlap_text + content

            # Add overlap to next chunk (handled in next iteration)
            overlapped_chunks.append(SemanticChunk(
                content=content,
                metadata={
                    **chunk.metadata,
                    "has_overlap": i > 0,
                    "overlap_chars": len(content) - len(chunk.content) if i > 0 else 0
                },
                chunk_id=chunk.chunk_id,
                semantic_score=chunk.semantic_score,
                boundary_type=chunk.boundary_type
            ))

        return overlapped_chunks


def get_semantic_chunker(**kwargs) -> AdvancedSemanticChunker:
    """Factory function to get semantic chunker"""
    return AdvancedSemanticChunker(**kwargs)
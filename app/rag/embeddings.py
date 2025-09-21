from abc import ABC, abstractmethod
from typing import List
import numpy as np

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from app.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.openai_api_key
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Using mock embeddings.")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing without API keys or heavy models"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        # Generate deterministic "embedding" based on text hash
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()

        # Convert hash to normalized vector
        vector = []
        for i in range(self.dimension):
            # Use hash bytes cyclically to create vector
            byte_val = hash_bytes[i % len(hash_bytes)]
            normalized_val = (byte_val / 255.0) * 2 - 1  # Scale to [-1, 1]
            vector.append(normalized_val)

        # Normalize vector to unit length
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


def get_embedding_provider() -> EmbeddingProvider:
    # Use mock embeddings if no API key or in debug mode
    use_mock = (
        not settings.openai_api_key or
        settings.debug or
        not OPENAI_AVAILABLE or
        not SENTENCE_TRANSFORMERS_AVAILABLE
    )

    if use_mock:
        return MockEmbeddingProvider()
    elif settings.openai_api_key and OPENAI_AVAILABLE:
        return OpenAIEmbeddingProvider()
    else:
        try:
            return LocalEmbeddingProvider()
        except ImportError:
            return MockEmbeddingProvider()
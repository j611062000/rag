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




def get_embedding_provider() -> EmbeddingProvider:
    # Priority: OpenAI API > Local embeddings > Error if neither available

    # First try OpenAI if API key is available
    if settings.openai_api_key and OPENAI_AVAILABLE:
        try:
            return OpenAIEmbeddingProvider()
        except Exception as e:
            print(f"OpenAI embeddings failed: {e}. Falling back to local embeddings.")

    # Then try local embeddings with sentence transformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return LocalEmbeddingProvider()
        except Exception as e:
            print(f"Local embeddings failed: {e}.")
            raise RuntimeError("No embedding provider available. Install sentence-transformers or provide OpenAI API key.")

    # No fallback - require proper embedding provider
    raise RuntimeError("No embedding provider available. Install sentence-transformers or provide OpenAI API key.")

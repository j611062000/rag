from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from dataclasses import dataclass
from loguru import logger

from app.config import settings
from app.rag.embeddings import get_embedding_provider


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        pass

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        pass

    @abstractmethod
    async def clear(self):
        pass

    @abstractmethod
    async def list_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        pass


class ChromaVectorStore(VectorStore):
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_provider = get_embedding_provider()


    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        # Process in batches for better memory management
        batch_size = 64

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            # Generate embeddings directly from original texts
            embeddings = self.embedding_provider.embed_documents(batch_texts)

            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        # Use raw query - let the better embedding model handle semantic understanding
        query_embedding = self.embedding_provider.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append(SearchResult(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1.0 - results['distances'][0][i]  # Convert distance to similarity
            ))

        return search_results

    async def clear(self):
        try:
            self.client.delete_collection("pdf_documents")
            self.collection = self.client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass

    async def list_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            result = self.collection.get(limit=limit)
            documents = []

            if result['documents']:
                for i, doc in enumerate(result['documents']):
                    documents.append({
                        'id': result['ids'][i] if result['ids'] else f"doc_{i}",
                        'content': doc,
                        'metadata': result['metadatas'][i] if result['metadatas'] else {},
                        'content_preview': doc[:200] + "..." if len(doc) > 200 else doc
                    })

            return documents
        except Exception as e:
            logger.error(f"Error listing documents from ChromaDB: {str(e)}")
            return []



def get_vector_store() -> VectorStore:
    if settings.vector_db == "chroma":
        return ChromaVectorStore()
    else:
        raise RuntimeError("vector db init error")

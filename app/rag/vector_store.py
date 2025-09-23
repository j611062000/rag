from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
import faiss
import numpy as np
import pickle
import os
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
        embeddings = self.embedding_provider.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
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


class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.embedding_provider = get_embedding_provider()
        self.documents = []
        self.metadatas = []

        # Try to load existing index
        if os.path.exists(f"{index_path}.index") and os.path.exists(f"{index_path}_metadata.pkl"):
            self.index = faiss.read_index(f"{index_path}.index")
            with open(f"{index_path}_metadata.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
        else:
            self.index = None

    async def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        embeddings = np.array(self.embedding_provider.embed_documents(texts)).astype('float32')

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Save to disk
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}_metadata.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = np.array([self.embedding_provider.embed_text(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))

        search_results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                search_results.append(SearchResult(
                    content=self.documents[idx],
                    metadata=self.metadatas[idx],
                    score=float(score)
                ))

        return search_results

    async def clear(self):
        if os.path.exists(f"{self.index_path}.index"):
            os.remove(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}_metadata.pkl"):
            os.remove(f"{self.index_path}_metadata.pkl")

        self.index = None
        self.documents = []
        self.metadatas = []

    async def list_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        documents = []
        total_docs = len(self.documents)

        if total_docs == 0:
            return documents

        end_idx = min(limit, total_docs) if limit else total_docs

        for i in range(end_idx):
            documents.append({
                'id': f"faiss_{i}",
                'content': self.documents[i],
                'metadata': self.metadatas[i],
                'content_preview': self.documents[i][:200] + "..." if len(self.documents[i]) > 200 else self.documents[i]
            })

        return documents


def get_vector_store() -> VectorStore:
    if settings.vector_db == "chroma":
        return ChromaVectorStore()
    else:
        return FAISSVectorStore()
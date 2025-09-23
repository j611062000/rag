from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from dataclasses import dataclass
from loguru import logger

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

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
        # Validate inputs
        if not texts or len(texts) == 0:
            logger.warning("No texts provided to add_documents")
            return

        if len(texts) != len(metadatas) or len(texts) != len(ids):
            logger.error(f"Mismatched lengths: texts={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)}")
            raise ValueError("texts, metadatas, and ids must have the same length")

        # Process in batches for better memory management
        batch_size = 64

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            try:
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} documents")

                # Filter out empty texts
                valid_items = [(t, m, id_val) for t, m, id_val in zip(batch_texts, batch_metadatas, batch_ids) if t.strip()]

                if not valid_items:
                    logger.warning(f"Batch {i//batch_size + 1} has no valid texts, skipping")
                    continue

                valid_texts, valid_metadatas, valid_ids = zip(*valid_items)
                valid_texts = list(valid_texts)
                valid_metadatas = list(valid_metadatas)
                valid_ids = list(valid_ids)

                # Generate embeddings directly from original texts
                logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
                embeddings = self.embedding_provider.embed_documents(valid_texts)

                if len(embeddings) != len(valid_texts):
                    logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(valid_texts)} texts")
                    continue

                logger.debug(f"Adding {len(valid_texts)} documents to collection")
                self.collection.add(
                    documents=valid_texts,
                    embeddings=embeddings,
                    metadatas=valid_metadatas,
                    ids=valid_ids
                )

            except Exception as batch_e:
                logger.error(f"Error in batch {i//batch_size + 1}: {str(batch_e)}")
                raise

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



class VectorStoreRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper for our VectorStore"""

    def __init__(self, vector_store: VectorStore, search_kwargs: Optional[Dict] = None):
        super().__init__()
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 10}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using our vector store and convert to LangChain format"""
        k = self.search_kwargs.get("k", 10)

        # Use our existing async search method (need to run in sync context)
        import asyncio
        import nest_asyncio

        try:
            # Handle nested event loops
            nest_asyncio.apply()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            search_results = loop.run_until_complete(
                self.vector_store.search(query, k=k)
            )
            loop.close()
        except Exception as e:
            logger.error(f"Error in retriever search: {str(e)}")
            search_results = []

        # Convert SearchResult objects to LangChain Document objects
        documents = []
        for result in search_results:
            doc = Document(
                page_content=result.content,
                metadata={
                    **result.metadata,
                    "score": result.score  # Add score to metadata
                }
            )
            documents.append(doc)

        return documents

def get_vector_store() -> VectorStore:
    if settings.vector_db == "chroma":
        return ChromaVectorStore()
    else:
        raise RuntimeError("vector db init error")

def get_retriever(search_kwargs: Optional[Dict] = None) -> VectorStoreRetriever:
    """Get a LangChain-compatible retriever"""
    vector_store = get_vector_store()
    return VectorStoreRetriever(vector_store, search_kwargs)

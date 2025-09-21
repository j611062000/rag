from typing import Dict, Any, List
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse
from app.rag.vector_store import get_vector_store, SearchResult
from app.config import settings


class PDFRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.vector_store = get_vector_store()
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "retrieved_docs"],
            template="""
You are a PDF RAG agent that answers questions based on retrieved document chunks from academic papers.

Context from previous conversation:
{context}

User Question: {question}

Retrieved Documents:
{retrieved_docs}

Instructions:
1. Use ONLY the information from the retrieved documents to answer the question
2. If the retrieved documents don't contain enough information, say so clearly
3. Cite specific sources when possible (mention document names, page numbers if available)
4. Be precise and academic in your tone
5. If multiple documents provide different perspectives, acknowledge this

Provide your answer based solely on the retrieved information:
"""
        )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        # Retrieve relevant documents
        search_results = await self.vector_store.search(
            question,
            k=settings.max_retrieval_results
        )

        if not search_results:
            return AgentResponse(
                content="I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded the relevant PDF documents first.",
                metadata={
                    "sources": [],
                    "retrieved_chunks": 0
                },
                confidence=0.1
            )

        # Format retrieved documents
        retrieved_docs = self._format_search_results(search_results)

        # Generate response
        prompt_text = self.prompt.format(
            question=question,
            context=context,
            retrieved_docs=retrieved_docs
        )

        response = await self.llm.ainvoke(prompt_text)

        # Extract source information
        sources = self._extract_sources(search_results)

        return AgentResponse(
            content=response.content,
            metadata={
                "sources": sources,
                "retrieved_chunks": len(search_results),
                "search_results": [
                    {
                        "content": result.content[:200] + "...",
                        "score": result.score,
                        "metadata": result.metadata
                    } for result in search_results
                ]
            },
            confidence=min(0.9, sum(r.score for r in search_results) / len(search_results))
        )

    def _format_search_results(self, search_results: List[SearchResult]) -> str:
        formatted_docs = []

        for i, result in enumerate(search_results, 1):
            metadata = result.metadata
            source_info = f"Source {i}"

            if metadata.get("filename"):
                source_info += f" - {metadata['filename']}"
            if metadata.get("chunk_index") is not None:
                source_info += f" (Chunk {metadata['chunk_index']})"

            formatted_docs.append(f"{source_info}:\n{result.content}\n")

        return "\n".join(formatted_docs)

    def _extract_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        sources = []
        seen_documents = set()

        for result in search_results:
            metadata = result.metadata
            doc_id = metadata.get("document_id", "unknown")

            if doc_id not in seen_documents:
                sources.append({
                    "document_id": doc_id,
                    "filename": metadata.get("filename", "Unknown"),
                    "relevance_score": result.score
                })
                seen_documents.add(doc_id)

        return sources
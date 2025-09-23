"""
Advanced query processing pipeline using LlamaIndex
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

try:
    from llama_index.core import QueryBundle
    from llama_index.core.query_pipeline import (
        QueryPipeline,
        InputComponent,
        FnComponent
    )
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.response_synthesizers import ResponseMode
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

from loguru import logger
from app.agents.base import BaseAgent, AgentResponse
from app.rag.vector_store import get_vector_store


@dataclass
class QueryResult:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_steps: List[str]
    query_transformations: List[str]


class LlamaIndexRetrieverWrapper:
    """Wrapper to make our vector store compatible with LlamaIndex"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def retrieve(self, query_bundle) -> List[Dict]:
        """Retrieve documents for LlamaIndex"""
        query_str = str(query_bundle.query_str)

        # Use our existing search
        search_results = await self.vector_store.search(query_str, k=10)

        # Convert to LlamaIndex format
        nodes = []
        for result in search_results:
            node = {
                "text": result.content,
                "metadata": result.metadata,
                "score": result.score
            }
            nodes.append(node)

        return nodes


class AdvancedQueryPipeline:
    """Sophisticated query processing using LlamaIndex pipelines"""

    def __init__(self, llm):
        self.llm = llm
        self.vector_store = get_vector_store()
        self.processing_steps = []

        if not LLAMA_INDEX_AVAILABLE:
            logger.warning("LlamaIndex not available, using simplified pipeline")
            self.use_advanced_pipeline = False
        else:
            self.use_advanced_pipeline = True
            self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup the LlamaIndex query pipeline"""
        try:
            # Query transformation component
            def enhance_query(query_str: str) -> Dict[str, str]:
                """Enhance the query with context and reformulations"""
                self.processing_steps.append("query_enhancement")

                # Simple query enhancement (you can make this more sophisticated)
                enhanced_queries = {
                    "original": query_str,
                    "focused": self._create_focused_query(query_str),
                    "broad": self._create_broad_query(query_str),
                    "technical": self._create_technical_query(query_str)
                }

                return enhanced_queries

            # Retrieval component
            def multi_retrieve(enhanced_queries: Dict[str, str]) -> List[Dict]:
                """Retrieve using multiple query variations"""
                self.processing_steps.append("multi_retrieval")

                all_results = []
                retriever = LlamaIndexRetrieverWrapper(self.vector_store)

                for query_type, query in enhanced_queries.items():
                    try:
                        # Create query bundle
                        query_bundle = QueryBundle(query_str=query)

                        # Retrieve documents (this needs to be sync for LlamaIndex)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(retriever.retrieve(query_bundle))
                        loop.close()

                        # Tag results with query type
                        for result in results:
                            result["query_type"] = query_type

                        all_results.extend(results)

                    except Exception as e:
                        logger.error(f"Retrieval failed for {query_type}: {str(e)}")
                        continue

                return all_results

            # Filtering and ranking component
            def filter_and_rank(results: List[Dict]) -> List[Dict]:
                """Filter and rank results"""
                self.processing_steps.append("filtering_ranking")

                if not results:
                    return []

                # Remove duplicates
                unique_results = []
                seen_content = set()

                for result in results:
                    content_hash = hash(result["text"][:100])  # Use first 100 chars as identifier
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_results.append(result)

                # Sort by score
                ranked_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)

                # Take top results and apply quality threshold
                quality_threshold = 0.3
                filtered_results = [r for r in ranked_results[:15] if r.get("score", 0) >= quality_threshold]

                return filtered_results

            # Response synthesis component
            def synthesize_response(filtered_results: List[Dict], original_query: str) -> Dict[str, Any]:
                """Synthesize final response"""
                self.processing_steps.append("response_synthesis")

                if not filtered_results:
                    return {
                        "answer": "I couldn't find relevant information to answer your question.",
                        "confidence": 0.1,
                        "sources": []
                    }

                # Prepare context for LLM
                context_pieces = []
                sources = []

                for i, result in enumerate(filtered_results):
                    context_pieces.append(f"Source {i+1}: {result['text']}")
                    sources.append({
                        "content": result["text"][:200] + "...",
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0),
                        "query_type": result.get("query_type", "unknown")
                    })

                context = "\n\n".join(context_pieces)

                # Generate response using LLM
                synthesis_prompt = f"""
Based on the following sources, provide a comprehensive answer to the user's question.

Question: {original_query}

Sources:
{context}

Instructions:
1. Answer directly and precisely
2. Cite specific sources when making claims
3. If sources contain conflicting information, acknowledge this
4. If information is insufficient, state what's missing
5. Focus on factual accuracy over completeness

Answer:"""

                try:
                    # This would need to be adapted for your LLM interface
                    response = asyncio.run(self.llm.ainvoke(synthesis_prompt))
                    answer = response.content

                    # Calculate confidence based on source quality
                    avg_score = sum(r.get("score", 0) for r in filtered_results) / len(filtered_results)
                    confidence = min(0.9, avg_score + 0.1)  # Add small boost

                    return {
                        "answer": answer,
                        "confidence": confidence,
                        "sources": sources
                    }

                except Exception as e:
                    logger.error(f"Response synthesis failed: {str(e)}")
                    return {
                        "answer": "I encountered an error while synthesizing the response.",
                        "confidence": 0.1,
                        "sources": sources
                    }

            # Build the pipeline
            self.pipeline = QueryPipeline()

            # Add components
            self.pipeline.add_modules({
                "input": InputComponent(),
                "enhance": FnComponent(fn=enhance_query),
                "retrieve": FnComponent(fn=multi_retrieve),
                "filter": FnComponent(fn=filter_and_rank),
                "synthesize": FnComponent(fn=synthesize_response, req_keys=["filtered_results", "original_query"])
            })

            # Connect the pipeline
            self.pipeline.add_link("input", "enhance")
            self.pipeline.add_link("enhance", "retrieve")
            self.pipeline.add_link("retrieve", "filter")
            self.pipeline.add_link("filter", "synthesize", dest_key="filtered_results")
            self.pipeline.add_link("input", "synthesize", dest_key="original_query")

            logger.info("Advanced query pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup advanced pipeline: {str(e)}")
            self.use_advanced_pipeline = False

    async def process_query(self, question: str, context: str = "") -> QueryResult:
        """Process query through advanced pipeline"""
        self.processing_steps = []  # Reset steps

        if self.use_advanced_pipeline:
            return await self._process_with_pipeline(question, context)
        else:
            return await self._process_simple(question, context)

    async def _process_with_pipeline(self, question: str, context: str) -> QueryResult:
        """Process using LlamaIndex pipeline"""
        try:
            # Add context to question if available
            enhanced_question = f"{context}\n\nQuestion: {question}" if context else question

            # Run the pipeline
            result = self.pipeline.run(input=enhanced_question)

            # Extract transformations used
            query_transformations = [
                "original query",
                "focused variation",
                "broad variation",
                "technical variation"
            ]

            return QueryResult(
                answer=result.get("answer", "No answer generated"),
                sources=result.get("sources", []),
                confidence=result.get("confidence", 0.5),
                processing_steps=self.processing_steps.copy(),
                query_transformations=query_transformations
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return await self._process_simple(question, context)

    async def _process_simple(self, question: str, context: str) -> QueryResult:
        """Fallback simple processing"""
        try:
            # Simple retrieval
            search_results = await self.vector_store.search(question, k=5)

            if not search_results:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.1,
                    processing_steps=["simple_search"],
                    query_transformations=["original_query"]
                )

            # Simple synthesis
            context_pieces = [f"Source: {result.content}" for result in search_results]
            context_text = "\n\n".join(context_pieces)

            prompt = f"""
Based on the following information, answer the question.

Question: {question}
Context: {context}

Information:
{context_text}

Answer:"""

            response = await self.llm.ainvoke(prompt)

            sources = [
                {
                    "content": result.content[:200] + "...",
                    "metadata": result.metadata,
                    "score": result.score
                }
                for result in search_results
            ]

            confidence = sum(r.score for r in search_results) / len(search_results)

            return QueryResult(
                answer=response.content,
                sources=sources,
                confidence=confidence,
                processing_steps=["simple_search", "simple_synthesis"],
                query_transformations=["original_query"]
            )

        except Exception as e:
            logger.error(f"Simple processing failed: {str(e)}")
            return QueryResult(
                answer="I encountered an error while processing your question.",
                sources=[],
                confidence=0.1,
                processing_steps=["error"],
                query_transformations=[]
            )

    def _create_focused_query(self, query: str) -> str:
        """Create a more focused version of the query"""
        # Simple focused query generation
        return f"specific details about {query}"

    def _create_broad_query(self, query: str) -> str:
        """Create a broader version of the query"""
        # Simple broad query generation
        return f"overview context information {query}"

    def _create_technical_query(self, query: str) -> str:
        """Create a technical version of the query"""
        # Simple technical query generation
        return f"technical details results data {query}"


class AdvancedPDFAgent(BaseAgent):
    """PDF agent using advanced query pipeline"""

    def __init__(self):
        super().__init__()
        self.query_pipeline = AdvancedQueryPipeline(self.llm)

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        # Use advanced pipeline
        result = await self.query_pipeline.process_query(question, context)

        return AgentResponse(
            content=result.answer,
            metadata={
                "sources": [{"filename": s["metadata"].get("filename", "Unknown")} for s in result.sources],
                "retrieved_chunks": len(result.sources),
                "processing_steps": result.processing_steps,
                "query_transformations": result.query_transformations,
                "pipeline_method": "llamaindex_advanced"
            },
            confidence=result.confidence
        )


def get_advanced_pdf_agent() -> AdvancedPDFAgent:
    """Factory function to create advanced PDF agent"""
    return AdvancedPDFAgent()
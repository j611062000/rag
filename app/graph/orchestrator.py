from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger

from app.agents.base import AgentResponse
from app.agents.clarifier import ClarificationAgent
from app.agents.router import RoutingAgent
from app.agents.pdf_agent import PDFRAGAgent
from app.agents.web_agent import WebSearchAgent
from app.agents.synthesizer import AnswerSynthesizer
from app.memory.session import SessionManager


@dataclass
class QueryState:
    question: str
    session_id: str
    context: str = ""
    is_clear: bool = True
    route: str = ""
    pdf_result: AgentResponse = None
    web_result = None
    final_answer: str = ""
    sources: List[Dict[str, Any]] = None
    confidence: float = 0.0


class ChatOrchestrator:
    def __init__(self):
        self.clarifier = ClarificationAgent()
        self.router = RoutingAgent()
        self.pdf_agent = PDFRAGAgent()
        self.web_agent = WebSearchAgent()
        self.synthesizer = AnswerSynthesizer()
        self.session_manager = SessionManager()

    async def process_query(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        # Initialize state
        state = QueryState(
            question=question,
            session_id=session_id,
            sources=[]
        )

        try:
            # Get conversation context
            state.context = await self.session_manager.get_context(session_id)

            # Step 1: Clarification
            logger.info("Step 1: Checking question clarity")
            clarification_result = await self.clarifier.process({
                "question": question,
                "context": state.context
            })

            state.is_clear = clarification_result.metadata.get("is_clear", True)

            if not state.is_clear:
                # Return clarification request
                await self.session_manager.store_message(
                    session_id, "question", question
                )
                await self.session_manager.store_message(
                    session_id, "clarification", clarification_result.content
                )

                return {
                    "answer": f"I need some clarification: {clarification_result.content}",
                    "sources": [],
                    "confidence": 0.3,
                    "needs_clarification": True
                }

            # Step 2: Routing
            logger.info("Step 2: Determining routing strategy")
            routing_result = await self.router.process({
                "question": question,
                "context": state.context
            })

            state.route = routing_result.metadata.get("route", "web")
            logger.info(f"Routing decision: {state.route}")

            # Step 3: Information Retrieval with Fallback Logic
            if state.route == "pdf":
                state.pdf_result = await self.pdf_agent.process({
                    "question": question,
                    "context": state.context
                })
                logger.info(f"PDF retrieval completed with confidence {state.pdf_result.confidence}")

                # Fallback to web search if PDF results are insufficient
                should_fallback = (
                    state.pdf_result.confidence < 0.5 or
                    state.pdf_result.metadata.get("retrieved_chunks", 0) == 0 or
                    "I couldn't find any relevant information" in state.pdf_result.content or
                    "don't contain enough information" in state.pdf_result.content
                )

                if should_fallback:
                    logger.info(f"PDF results insufficient (confidence: {state.pdf_result.confidence}, chunks: {state.pdf_result.metadata.get('retrieved_chunks', 0)}), falling back to web search")
                    state.web_result = await self.web_agent.process({
                        "question": question,
                        "context": state.context
                    })
                    logger.info(f"Web search fallback completed with confidence {state.web_result.confidence}")
                    # Update route to indicate fallback occurred
                    state.route = "pdf_with_web_fallback"

            elif state.route == "web":
                state.web_result = await self.web_agent.process({
                    "question": question,
                    "context": state.context
                })
                logger.info(f"Web search completed with confidence {state.web_result.confidence}")

            elif state.route == "both":
                # Run both in parallel
                state.pdf_result = await self.pdf_agent.process({
                    "question": question,
                    "context": state.context
                })
                state.web_result = await self.web_agent.process({
                    "question": question,
                    "context": state.context
                })
                logger.info(f"Both retrieval completed. PDF: {state.pdf_result.confidence}, Web: {state.web_result.confidence}")

            # Step 4: Answer Synthesis
            logger.info("Step 4: Synthesizing final answer")
            synthesis_result = await self.synthesizer.process({
                "question": question,
                "context": state.context,
                "pdf_result": state.pdf_result,
                "web_result": state.web_result
            })

            state.final_answer = synthesis_result.content
            state.sources = synthesis_result.metadata.get("sources", [])
            state.confidence = synthesis_result.confidence

            # Store in session memory
            await self.session_manager.store_message(
                session_id, "question", question
            )
            await self.session_manager.store_message(
                session_id, "answer", state.final_answer,
                {
                    "sources": state.sources,
                    "confidence": state.confidence,
                    "route": state.route
                }
            )

            logger.info(f"Query processing completed with confidence {state.confidence}")

            return {
                "answer": state.final_answer,
                "sources": state.sources,
                "confidence": state.confidence,
                "route_used": state.route,
                "needs_clarification": False,
                "used_pdf": synthesis_result.metadata.get("used_pdf", False),
                "used_web": synthesis_result.metadata.get("used_web", False),
                "pdf_confidence": synthesis_result.metadata.get("pdf_confidence", 0.0),
                "web_confidence": synthesis_result.metadata.get("web_confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

            # Store error in session for context
            await self.session_manager.store_message(
                session_id, "question", question
            )
            await self.session_manager.store_message(
                session_id, "error", f"Processing error: {str(e)}"
            )

            return {
                "answer": "I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
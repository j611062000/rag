from typing import Dict, Any, List
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse


class AnswerSynthesizer(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "pdf_result", "web_result"],
            template="""
You are an answer synthesizer that combines information from PDF documents and web search to provide comprehensive answers.

Context from previous conversation:
{context}

User Question: {question}

PDF RAG Result:
{pdf_result}

Web Search Result:
{web_result}

Instructions:
1. Synthesize information from both sources to provide a complete answer
2. ALWAYS clearly indicate the source of information using these formats:
   - "According to the PDF documents..." or "From the uploaded papers..."
   - "Based on web search results..." or "From web sources..."
3. Use clear attribution phrases like:
   - "📄 **From PDF:** [information from documents]"
   - "🌐 **From Web:** [information from web search]"
4. Highlight any contradictions or different perspectives between sources
5. Provide specific citations when available
6. Give preference to academic sources for theoretical concepts
7. Use web sources for current information, practical applications, or general context

Structure your response with clear source attribution throughout:
"""
        )

        self.single_source_prompt = PromptTemplate(
            input_variables=["question", "context", "source_result", "source_type", "source_icon"],
            template="""
You are providing an answer based on {source_type} information.

Context from previous conversation:
{context}

User Question: {question}

{source_type} Result:
{source_result}

Instructions:
1. ALWAYS start your response by clearly indicating the source
2. Use the format: "{source_icon} **From {source_type}:** [your answer]"
3. Be comprehensive but clearly attribute all information to the {source_type}
4. Include specific citations when available

Provide a clear, comprehensive answer with proper source attribution:
"""
        )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        pdf_result = input_data.get("pdf_result")
        web_result = input_data.get("web_result")

        # Determine which sources are available
        has_pdf = pdf_result and pdf_result.content and not self._is_no_result(pdf_result.content)
        has_web = web_result and web_result.content and not self._is_no_result(web_result.content)

        if has_pdf and has_web:
            # Synthesize both sources
            prompt_text = self.prompt.format(
                question=question,
                context=context,
                pdf_result=pdf_result.content,
                web_result=web_result.content
            )
            sources = []
            if pdf_result.metadata and pdf_result.metadata.get("sources"):
                sources.extend(pdf_result.metadata["sources"])
            if web_result.metadata and web_result.metadata.get("sources"):
                sources.extend(web_result.metadata["sources"])

            confidence = (pdf_result.confidence + web_result.confidence) / 2

        elif has_pdf:
            # Use only PDF source
            prompt_text = self.single_source_prompt.format(
                question=question,
                context=context,
                source_result=pdf_result.content,
                source_type="PDF documents",
                source_icon="📄"
            )
            sources = pdf_result.metadata.get("sources", []) if pdf_result.metadata else []
            confidence = pdf_result.confidence

        elif has_web:
            # Use only web source
            prompt_text = self.single_source_prompt.format(
                question=question,
                context=context,
                source_result=web_result.content,
                source_type="web search",
                source_icon="🌐"
            )
            sources = web_result.metadata.get("sources", []) if web_result.metadata else []
            confidence = web_result.confidence

        else:
            # No valid sources
            return AgentResponse(
                content="I couldn't find relevant information from either the uploaded documents or web search to answer your question. Please ensure you've uploaded relevant PDF documents or check if your question can be answered with available resources.",
                metadata={"sources": []},
                confidence=0.1
            )

        # Generate synthesized response
        response = await self.llm.ainvoke(prompt_text)

        return AgentResponse(
            content=response.content,
            metadata={
                "sources": sources,
                "used_pdf": has_pdf,
                "used_web": has_web,
                "pdf_confidence": pdf_result.confidence if pdf_result else 0.0,
                "web_confidence": web_result.confidence if web_result else 0.0
            },
            confidence=confidence
        )

    def _is_no_result(self, content: str) -> bool:
        """Check if the content indicates no results were found"""
        no_result_phrases = [
            "couldn't find",
            "no relevant information",
            "don't contain enough information",
            "unable to find",
            "no information available"
        ]
        return any(phrase in content.lower() for phrase in no_result_phrases)
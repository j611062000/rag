from typing import Dict, Any
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse
from app.search.web_search import get_web_search_provider


class WebSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.search_provider = get_web_search_provider()
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "search_results"],
            template="""
You are a web search agent that answers questions using information retrieved from web search.

Context from previous conversation:
{context}

User Question: {question}

Search Results:
{search_results}

Instructions:
1. Use the web search results to provide a comprehensive answer
2. Synthesize information from multiple sources when available
3. Cite sources with their URLs
4. If the search results don't contain relevant information, say so clearly
5. Focus on factual, current information

Provide your answer based on the search results:
"""
        )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        # Perform web search
        search_results = await self.search_provider.search(question, max_results=5)

        if not search_results:
            return AgentResponse(
                content="I couldn't find any relevant information from web search to answer your question. Please try rephrasing your question or check your internet connection.",
                metadata={
                    "sources": [],
                    "search_results_count": 0
                },
                confidence=0.1
            )

        # Format search results
        formatted_results = self._format_search_results(search_results)

        # Generate response
        prompt_text = self.prompt.format(
            question=question,
            context=context,
            search_results=formatted_results
        )

        response = await self.llm.ainvoke(prompt_text)

        # Extract source information
        sources = [
            {
                "title": result.title,
                "url": result.url,
                "score": result.score
            }
            for result in search_results
        ]

        return AgentResponse(
            content=response.content,
            metadata={
                "sources": sources,
                "search_results_count": len(search_results),
                "search_query": question
            },
            confidence=min(0.8, sum(r.score for r in search_results) / len(search_results)) if search_results else 0.1
        )

    def _format_search_results(self, search_results) -> str:
        formatted_results = []

        for i, result in enumerate(search_results, 1):
            formatted_results.append(
                f"Source {i} - {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.content[:500]}...\n"
            )

        return "\n".join(formatted_results)
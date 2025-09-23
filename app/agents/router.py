from typing import Dict, Any
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse


class RoutingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a routing agent that decides whether to search PDF documents or the web.

Context from previous conversation:
{context}

User Question: {question}

ROUTING RULES:
1. ALWAYS use "PDF" by default for any question
2. ONLY use "WEB" if the user explicitly asks for web search, online search, or internet search
   - Examples: "search the web for...", "look online for...", "find on the internet..."
3. ONLY use "BOTH" if the user explicitly asks to compare PDF content with web results

The system has automatic fallback: if PDF search doesn't find good results, it will automatically search the web.
So always prefer PDF unless explicitly requested otherwise.

Check if the user question contains explicit web search requests:
- Words like: "web", "online", "internet", "search online", "look up online", "google", "browse"
- Phrases like: "search the web", "find online", "look on the internet"

If NO explicit web search request is found, always use PDF.

Respond with exactly one of:
- "PDF: Searching PDF documents first (automatic web fallback if needed)"
- "WEB: User explicitly requested web/online search"
- "BOTH: User requested comparing PDF and web results"

Your response:
"""
        )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        prompt_text = self.prompt.format(question=question, context=context)
        response = await self.llm.ainvoke(prompt_text)

        content = response.content.strip()

        # Parse the routing decision
        if content.startswith("PDF:"):
            route = "pdf"
            reason = content[4:].strip()
        elif content.startswith("WEB:"):
            route = "web"
            reason = content[4:].strip()
        elif content.startswith("BOTH:"):
            route = "both"
            reason = content[5:].strip()
        else:
            # Default fallback to PDF search
            route = "pdf"
            reason = "Defaulting to PDF search for academic question."

        return AgentResponse(
            content=reason,
            metadata={
                "route": route,
                "original_question": question
            },
            confidence=0.9
        )
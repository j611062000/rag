from typing import Dict, Any
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse


class RoutingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a routing agent that decides whether a question should be answered using:
1. PDF documents (internal knowledge base)
2. Web search (external information)
3. Both (requires information from both sources)

Context from previous conversation:
{context}

User Question: {question}

Analyze the question and determine the best approach:

- Use "PDF" if the question is about:
  * Academic papers, research findings, specific studies
  * Technical concepts that would be in academic literature
  * Citations, authors, methodologies from papers
  * Specific results, experiments, or data from papers
  * Questions that are likely to be answered by documents you've already ingested

- Use "WEB" if the question is about:
  * Current events, news, recent developments
  * General knowledge not typically found in academic papers
  * Practical applications, tutorials, how-to guides
  * Company information, product details
  * Questions clearly outside the scope of academic PDFs

- Use "BOTH" if the question requires:
  * Comparing academic findings with current information
  * Contextualizing research within current trends
  * Both theoretical background and practical applications

Note: If PDF search doesn't find sufficient information (low confidence), the system will automatically fallback to web search, so prefer PDF when in doubt for academic questions.

Respond with exactly one of:
- "PDF: <reason for using PDF search>"
- "WEB: <reason for using web search>"
- "BOTH: <reason for using both sources>"

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
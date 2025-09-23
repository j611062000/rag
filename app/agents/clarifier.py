from typing import Dict, Any
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse


class ClarificationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a clarification agent. Your job is to determine if a user's question is clear and specific enough to answer, or if it needs clarification.

Only ask for clarification if the question is truly ambiguous, incomplete, or lacks essential context.

Context from previous conversation:
{context}

User Question: {question}

Guidelines for when to consider a question CLEAR:
- Questions about paper contributions, findings, or results are usually clear
- Questions referencing specific papers by title or authors are clear
- Questions asking for specific information from documents are clear
- General questions about research topics are clear

Only require clarification for:
- Extremely vague questions with no context
- Questions with multiple possible interpretations
- Questions where the user asks "what about this?" without specifying what "this" is

Respond with either:
- "CLEAR: <reason why it's clear>"
- "NEEDS_CLARIFICATION: <what specific clarification is needed>"

Your response:
"""
        )

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        question = input_data.get("question", "")
        context = input_data.get("context", "")

        prompt_text = self.prompt.format(question=question, context=context)
        response = await self.llm.ainvoke(prompt_text)

        content = response.content.strip()

       
        if content.startswith("NEEDS_CLARIFICATION:"):
            is_clear = False
            reason = content[20:].strip()
        
        else:
            is_clear = True
            reason = content[6:].strip()

        return AgentResponse(
            content=reason,
            metadata={
                "is_clear": is_clear,
                "original_question": question
            },
            confidence=0.8
        )
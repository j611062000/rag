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

IMPORTANT: Be conservative about asking for clarification. Only ask when the question is genuinely ambiguous or impossible to answer.

Context from previous conversation:
{context}

User Question: {question}

Questions that are CLEAR and should NOT require clarification:
- Technical questions with specific references (papers, datasets, frameworks)
- Questions about specific entities, companies, or well-known concepts
- Follow-up questions that reference previous conversation context
- Questions with sufficient detail to understand the user's intent
- Academic questions citing authors, years, or specific studies

Only require clarification for:
- Extremely vague questions like "What about this?" with no referent
- Questions where key information is completely missing
- Ambiguous pronouns without clear antecedents when context doesn't help
- Questions that could have multiple completely different interpretations

Examples of questions that are CLEAR:
- "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?"
- "What are the main features of React hooks?"
- "How does transformer attention work?"
- "What did the previous study conclude about X?"

Examples requiring clarification:
- "What about that thing we discussed?" (no context about what "thing")
- "How do I fix this?" (no indication what "this" refers to)
- "What's the best approach?" (no context about approach to what)

Given the context and question, respond with either:
- "CLEAR: <brief reason why it's clear>"
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
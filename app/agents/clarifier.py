from typing import Dict, Any
from langchain.prompts import PromptTemplate

from app.agents.base import BaseAgent, AgentResponse


class ClarificationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a clarification agent that determines if a user's question is clear enough to answer, and if needed, combines it with previous conversation context.

Context from previous conversation:
{context}

Current User Question: {question}

Your tasks:
1. First, try to combine the current question with recent context to create a complete, standalone question
2. Then determine if clarification is still needed

## Context Combination Rules:

**Combine when the current question:**
- References unclear pronouns ("it", "this", "that", "the book", "the method", etc.)
- Is incomplete but previous questions provide missing context
- Asks about something mentioned in recent conversation

**Examples of context combination:**

Previous: "What is machine learning?"
Current: "How does it work?"
→ Combined: "How does machine learning work?"

Previous: "Tell me about the book 'Clean Code'"
Current: "What is the main idea?"
→ Combined: "What is the main idea of the book 'Clean Code'?"

Previous: "Which algorithm is best for classification?"
Current: "What are its advantages?"
→ Combined: "What are the advantages of the best classification algorithm?"

## Response Format:

After attempting combination (if applicable), respond with either:
- "CLEAR: [brief reason why it's clear]"
- "NEEDS_CLARIFICATION: [what specific clarification is needed]"

If you combine the question with context, treat the combined question as the new question and evaluate whether it's clear.

## Guidelines:
- Only combine when it creates a meaningful, complete question
- Don't combine if it would change the user's intent
- Preserve the user's original phrasing style when possible
- Be conservative - better to ask for clarification than assume incorrectly

IMPORTANT: Use your judgment to determine if a question has enough context to provide a meaningful answer.

## Evaluation Criteria:

**Questions are CLEAR when they:**
- Have sufficient context to understand what the user is asking about
- Include enough specificity to provide a focused, helpful answer
- Reference concrete entities, concepts, or contexts
- Can be answered without making major assumptions about the user's intent

**Questions NEED CLARIFICATION when they:**
- Are missing key context that would significantly affect the answer
- Use vague or ambiguous terms without clear referents
- Could be interpreted in multiple fundamentally different ways
- Would require you to guess what the user is really asking about

## Your Task:
Evaluate whether the question, potentially combined with context, provides enough information to give a helpful, specific answer. If important details are missing that would change the nature of the response, ask for clarification about those specific missing pieces.

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
        else:  # CLEAR (handles both explicit "CLEAR:" and other responses)
            is_clear = True
            reason = content[6:].strip() if content.startswith("CLEAR:") else content

        metadata = {
            "is_clear": is_clear,
            "original_question": question
        }

        return AgentResponse(
            content=reason,
            metadata=metadata,
            confidence=0.8
        )
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from app.config import settings


@dataclass
class AgentResponse:
    content: str
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


class BaseAgent(ABC):
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain packages not available. Install with: pip install langchain-openai langchain-anthropic")

        # Prioritize Anthropic if available
        if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
            return ChatAnthropic(
                model=settings.llm_model,
                api_key=settings.anthropic_api_key,
                temperature=0.0
            )
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                temperature=0.0
            )
        else:
            raise ValueError("No valid LLM configuration found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        pass
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Literal
import os


def get_secret_from_env_or_file(env_var_name: str, fallback_value: str = "") -> str:
    """
    Get secret from OS environment variable first, then fall back to .env file value.
    This prioritizes OS env vars over .env file values for security.
    """
    # First try to get from OS environment
    os_value = os.getenv(env_var_name)
    if os_value and os_value not in ["your_anthropic_api_key_here", "your_openai_api_key_here", "your_tavily_api_key_here", "your_serpapi_key_here"]:
        return os_value

    # Fall back to .env file value if it's not a placeholder
    if fallback_value and fallback_value not in ["your_anthropic_api_key_here", "your_openai_api_key_here", "your_tavily_api_key_here", "your_serpapi_key_here"]:
        return fallback_value

    # Return empty string if no real key found
    return ""


class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "azure"] = "anthropic"
    llm_model: str = "claude-3-5-haiku-20241022"
    openai_api_key_raw: str = Field(default="", description="OpenAI API key from .env", alias="OPENAI_API_KEY")
    anthropic_api_key_raw: str = Field(default="", description="Anthropic API key from .env", alias="ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key, prioritizing OS environment variable"""
        return get_secret_from_env_or_file("OPENAI_API_KEY", self.openai_api_key_raw)

    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key, prioritizing OS environment variable"""
        return get_secret_from_env_or_file("ANTHROPIC_API_KEY", self.anthropic_api_key_raw)

    # Vector Database
    vector_db: Literal["chroma", "faiss"] = "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"

    # Web Search Configuration
    search_provider: Literal["tavily", "duckduckgo", "serpapi", "mock"] = "mock"
    tavily_api_key_raw: str = Field(default="", description="Tavily API key from .env", alias="TAVILY_API_KEY")
    serpapi_api_key_raw: str = Field(default="", description="SerpAPI key from .env", alias="SERPAPI_API_KEY")
    duckduckgo_enabled: bool = True

    @property
    def tavily_api_key(self) -> str:
        """Get Tavily API key, prioritizing OS environment variable"""
        return get_secret_from_env_or_file("TAVILY_API_KEY", self.tavily_api_key_raw)

    @property
    def serpapi_api_key(self) -> str:
        """Get SerpAPI key, prioritizing OS environment variable"""
        return get_secret_from_env_or_file("SERPAPI_API_KEY", self.serpapi_api_key_raw)

    # Application Settings
    debug: bool = True
    log_level: str = "INFO"
    max_chunk_size: int = 1000
    chunk_overlap: int = 400
    max_retrieval_results: int = 10

    class Config:
        env_file = ".env"


settings = Settings()

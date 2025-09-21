from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from app.config import settings


@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    score: float = 0.0


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        pass


class TavilySearchProvider(WebSearchProvider):
    def __init__(self):
        if not TavilyClient:
            raise ImportError("tavily-python not installed. Install with: pip install tavily-python")

        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not configured")

        self.client = TavilyClient(api_key=settings.tavily_api_key)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )

            results = []
            for item in response.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    url=item.get('url', ''),
                    score=item.get('score', 0.0)
                ))

            return results

        except Exception as e:
            print(f"Tavily search error: {e}")
            return []


class DuckDuckGoSearchProvider(WebSearchProvider):
    def __init__(self):
        if not DDGS:
            raise ImportError("duckduckgo-search not installed. Install with: pip install duckduckgo-search")

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))

            results = []
            for item in search_results:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    content=item.get('body', ''),
                    url=item.get('href', ''),
                    score=0.5  # DuckDuckGo doesn't provide scores
                ))

            return results

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class MockWebSearchProvider(WebSearchProvider):
    """Mock web search provider for testing and demonstration"""

    def __init__(self):
        # Mock responses for different query types
        self.mock_responses = {
            "machine learning": [
                SearchResult(
                    title="Machine Learning - Wikipedia",
                    content="Machine learning (ML) is a field of artificial intelligence (AI) that uses statistical techniques to give computer systems the ability to 'learn' from data, without being explicitly programmed. The term was coined in 1959 by Arthur Samuel.",
                    url="https://en.wikipedia.org/wiki/Machine_learning",
                    score=0.9
                ),
                SearchResult(
                    title="What is Machine Learning? | IBM",
                    content="Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn insights and recognize patterns from data, applying that learning to make increasingly better predictions.",
                    url="https://www.ibm.com/topics/machine-learning",
                    score=0.85
                )
            ],
            "artificial intelligence": [
                SearchResult(
                    title="Artificial Intelligence News - MIT Technology Review",
                    content="Latest developments in AI research, including breakthroughs in large language models, computer vision, and autonomous systems. Recent advances show promise for practical applications.",
                    url="https://www.technologyreview.com/topic/artificial-intelligence/",
                    score=0.88
                ),
                SearchResult(
                    title="AI Applications in Healthcare 2024",
                    content="Artificial intelligence is revolutionizing healthcare through diagnostic imaging, drug discovery, and personalized treatment plans. Recent studies show 40% improvement in diagnostic accuracy.",
                    url="https://www.healthcare-ai.com/applications-2024",
                    score=0.82
                )
            ],
            "default": [
                SearchResult(
                    title="Search Results for Your Query",
                    content="This is a mock web search result. In a real implementation, this would return actual web search results from providers like Tavily, DuckDuckGo, or Google. The content would be dynamically generated based on your search query.",
                    url="https://example.com/search-results",
                    score=0.7
                ),
                SearchResult(
                    title="Additional Information Source",
                    content="Mock search providers are useful for testing and development when you don't have API keys or want to avoid rate limits. This result demonstrates how multiple sources would be returned and ranked by relevance.",
                    url="https://example.com/additional-info",
                    score=0.65
                )
            ]
        }

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        # Simulate API delay
        import asyncio
        await asyncio.sleep(0.1)

        query_lower = query.lower()

        # Find best matching mock response
        for keyword, responses in self.mock_responses.items():
            if keyword in query_lower:
                return responses[:max_results]

        # Generate dynamic response for unknown queries
        dynamic_results = [
            SearchResult(
                title=f"Information about '{query}'",
                content=f"This is a mock search result for the query '{query}'. In a real implementation, this would contain relevant information from web sources about your specific question.",
                url=f"https://example.com/search?q={query.replace(' ', '+')}",
                score=0.75
            ),
            SearchResult(
                title=f"Related Topics to {query}",
                content=f"Additional context and related information about '{query}' would appear here. Mock results help test the system without requiring external API calls.",
                url=f"https://example.com/related/{query.replace(' ', '-')}",
                score=0.68
            )
        ]

        return dynamic_results[:max_results]


def get_web_search_provider() -> WebSearchProvider:
    # Check if we should use mock (no API keys or explicit mock setting)
    use_mock = (
        not settings.tavily_api_key or
        settings.search_provider == "mock" or
        settings.debug
    )

    if use_mock:
        return MockWebSearchProvider()
    elif settings.search_provider == "tavily" and settings.tavily_api_key:
        return TavilySearchProvider()
    elif settings.search_provider == "duckduckgo" or settings.duckduckgo_enabled:
        return DuckDuckGoSearchProvider()
    else:
        # Fallback to mock if nothing else is configured
        return MockWebSearchProvider()
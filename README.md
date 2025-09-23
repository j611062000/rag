📚 Chat-with-PDF Backend (Generative AI Assignment)

Overview

This project implements a backend system for intelligent Q&A over academic PDFs, with the ability to fall back on web search when answers are not found in the documents.

The system is built with a multi-agent architecture (via LangGraph), using Retrieval-Augmented Generation (RAG) to ground answers in the provided corpus.

⸻

✨ Features
	•	Question Answering over PDFs using embeddings + vector store
	•	Autonomous Multi-Agent Orchestration with LangGraph (clarification, routing, retrieval, synthesis)
	•	Web Search Integration (Tavily / DuckDuckGo / SerpAPI)
	•	Session-based Memory for contextual follow-ups
	•	REST API endpoints:
	•	POST /ask – ask a question
	•	POST /clear – clear session memory
	•	POST /ingest – ingest PDF papers

⸻

🏗️ High-Level Architecture

flowchart LR
    User[Client/CLI] --> API[FastAPI Service]

    subgraph API
      Router[Routing Agent]
      Clarifier[Clarification Agent]
      PDFAgent[PDF RAG Agent]
      WebAgent[Web Search Agent]
      Synth[Answer Synthesizer]
      Memory[Session Memory]
    end

    PDFAgent -->|retrieves| VS[(Vector Store)]
    PDFAgent -->|calls| LLM[LLM Provider]
    WebAgent --> Search[Search API]
    Synth --> LLM
    Synth --> Memory

Flow
	1.	User sends a query via /ask.
	2.	Clarification Agent checks if the query is vague.
	3.	Routing Agent decides whether to use PDF retrieval or web search.
	4.	PDF RAG Agent queries the vector DB and synthesizes answers with the LLM.
	5.	Web Search Agent retrieves external information when needed.
	6.	Answer Synthesizer combines results and responds.
	7.	Session Memory stores context for follow-up questions.

⸻

🛠️ Tech Stack
	•	Language: Python 3.11+
	•	Framework: FastAPI
	•	LLM Orchestration: LangGraph (preferred), LangChain as helper
	•	RAG Components:
	•	Vector DB: ChromaDB (default), switchable to FAISS/PGVector
	•	Embeddings: OpenAI text-embedding-3-large or local models (e.g., bge-large)
	•	PDF Parsing: pypdf, pdfminer.six
	•	LLM Providers: OpenAI / Anthropic / Azure OpenAI (configurable)
	•	Web Search: Tavily API (recommended), DuckDuckGo, SerpAPI
	•	Memory: Redis (session storage, caching)
	•	Containerization: Docker + docker-compose
	•	Logging/Tracing: loguru + OpenTelemetry + Prometheus metrics

⸻

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/${account}/chat-with-pdf.git
cd chat-with-pdf

2. Environment Setup

Create a .env file:

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here

VECTOR_DB=chroma
REDIS_URL=redis://redis:6379

SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your_tavily_key

3. Run with Docker

docker-compose up --build

This starts:
	•	api (FastAPI service)
	•	vector-db (ChromaDB)
	•	redis (session + cache)

4. API Endpoints
	•	POST /ingest → ingest PDFs
	•	POST /ask → ask a question
	•	POST /clear → clear session

Example request:

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What accuracy did davinci-codex achieve on Spider?"}'


⸻

📦 Project Structure

repo/
├─ app/
│  ├─ api/          # FastAPI routes
│  ├─ agents/       # Clarify, Router, RAG, Web, Synth
│  ├─ graph/        # LangGraph orchestration
│  ├─ rag/          # retrievers, chunkers, embeddings
│  ├─ memory/       # Redis session memory
│  ├─ search/       # Web search integrations
│  ├─ config.py
│  └─ main.py
├─ scripts/
│  └─ ingest_pdfs.py
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ tests/
│  └─ e2e_test.py
└─ README.md


⸻

🔮 Future Improvements
	•	Add evaluation system (golden Q&A pairs, confidence scoring)
	•	Use hybrid retrieval (BM25 + dense embeddings + cross-encoder reranker)
	•	Improve citation grounding (highlight page spans)
	•	Add streaming API for /ask
	•	Deploy at scale: load balancing, autoscaling, multi-tenant vector DB

⸻

✅ Deliverable Checklist
	•	Running container (docker-compose up)
	•	End-to-end pipeline: ingest PDF → ask → answer → clear memory
	•	Modular agent design with LangGraph
	•	README with architecture, run steps, improvements

⸻

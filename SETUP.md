# Chat-with-PDF Setup Guide

This is a proof of concept implementation of the Chat-with-PDF backend system as described in the README.

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: OpenAI API key for LLM and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For web search functionality
TAVILY_API_KEY=your_tavily_api_key_here

# The following settings can be left as defaults for testing
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
VECTOR_DB=chroma
REDIS_URL=redis://redis:6379
SEARCH_PROVIDER=tavily
```

### 2. Running with Docker (Recommended)

```bash
cd docker
docker-compose up --build
```

This will start:
- FastAPI service on http://localhost:8000
- ChromaDB on http://localhost:8001
- Redis on localhost:6379

### 3. Running Locally (Development)

Install dependencies:
```bash
pip install -r requirements.txt
```

Start services manually:
```bash
# Start Redis (in separate terminal)
redis-server

# Start ChromaDB (in separate terminal)
chroma run --host localhost --port 8001

# Update .env for local development
REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Start the API
python -m app.main
```

## Testing the System

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. Ingest a PDF

Using the API:
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

Using the ingestion script:
```bash
python scripts/ingest_pdfs.py path/to/your/document.pdf
# or for a directory
python scripts/ingest_pdfs.py path/to/pdf/directory/
```

### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'
```

### 4. Clear Session
```bash
curl -X POST "http://localhost:8000/clear" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "default"}'
```

## Example Workflow

1. **Start the system**: `cd docker && docker-compose up --build`
2. **Upload a research paper**: Use the `/ingest` endpoint
3. **Ask questions**: Use the `/ask` endpoint
4. **Follow-up questions**: The system maintains conversation context
5. **Clear when done**: Use the `/clear` endpoint

## API Endpoints

- `GET /health` - Health check
- `POST /ingest` - Upload and ingest PDF files
- `POST /ask` - Ask questions about the documents
- `POST /clear` - Clear session memory
- `GET /sessions/{session_id}/history` - Get conversation history

## Architecture Features Implemented

✅ **Multi-Agent System**: Clarification → Routing → Retrieval → Synthesis
✅ **PDF RAG**: Vector search with ChromaDB and OpenAI embeddings
✅ **Web Search**: Fallback to Tavily/DuckDuckGo when PDFs insufficient
✅ **Session Memory**: Redis-based conversation context
✅ **REST API**: FastAPI with async endpoints
✅ **Containerized**: Docker Compose setup with all services

## Troubleshooting

**Common Issues:**

1. **"No relevant information found"**: Make sure you've uploaded PDF documents first
2. **Redis connection error**: Ensure Redis is running (docker-compose handles this)
3. **ChromaDB connection error**: Check that ChromaDB service is healthy
4. **OpenAI API errors**: Verify your API key is valid and has credits

**Debug Mode:**

Set `DEBUG=true` in your `.env` file for detailed logging.

**Run Tests:**

```bash
# Make sure the API is running first
python tests/e2e_test.py
```

## Next Steps for Production

This POC demonstrates the core functionality. For production deployment, consider:

- Load balancing and horizontal scaling
- Persistent volume mounts for data
- Database migrations and backup strategies
- Rate limiting and authentication
- Monitoring and alerting setup
- CI/CD pipelines
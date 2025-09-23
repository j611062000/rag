# Repository Guidelines

## Project Structure & Module Organization
The backend application lives in `app/`, organized by concern: `app/api` exposes FastAPI routes, `app/agents` hosts LangGraph agents, `app/graph` wires the orchestration, `app/rag` handles retrievers and embeddings, and `app/memory` + `app/search` integrate Redis and external web search. Startup hooks and configuration sit in `app/startup.py` and `app/config.py`. Integration assets such as Docker manifests live in `docker/`, reusable utilities in `scripts/`, and end-to-end tests in `tests/`. OpenAPI exports are written to `openapi/openapi.json` on startup, and sample PDFs belong under `papers/` for local ingestion.

## Build, Test, and Development Commands
Install dependencies with:
```
pip install -r requirements.txt
```
Run the API locally after ensuring Redis and Chroma are available:
```
python -m app.main
```
For a full stack spin-up (API, Chroma, Redis), rely on Docker:
```
cd docker && docker-compose up --build
```
Use `start.sh` / `stop.sh` for guided orchestration and cleanup, especially when sharing the workflow with new contributors.

## Coding Style & Naming Conventions
Code targets Python 3.11+. Follow PEP 8 with four-space indentation, descriptive snake_case for modules, variables, and async coroutine names, and PascalCase for Pydantic models. Keep LangGraph node functions pure and annotate inputs/outputs with type hints. Log through `loguru` and prefer structured dict payloads for agent outputs. Store secrets in `.env`, never in source.

## Testing Guidelines
Tests rely on `pytest` and `pytest-asyncio`; the canonical suite is `python tests/e2e_test.py`, which expects the API to be running. Prefer async `httpx.AsyncClient` fixtures and name new tests `test_<feature>_<scenario>` for clarity. When adding agents, include workflow tests that assert both HTTP status and presence of `sources` in responses.

## Commit & Pull Request Guidelines
History favors short imperative subjects (e.g., `init`). Keep commits scoped to one logical change and include schema or API updates in the message body. Pull requests should describe intent, reference any tracked issue, list manual/automated test results, and attach sample `curl` transcripts or screenshots when behavior changes. Request review from another agent-owner before merging.

## Environment & Security Tips
Copy `.env.example` to `.env` and supply API keys for OpenAI, Anthropic, and Tavily as needed. Use mock keys for local development; never commit secrets. Ensure Docker Desktop is running before executing `docker-compose`, and verify health with `curl http://localhost:8000/health` post-deploy.

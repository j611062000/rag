import asyncio
import httpx
import pytest
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


class TestChatWithPDFE2E:
    base_url = "http://localhost:8000"

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=self.base_url)

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test that the API is running"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_ask_without_documents(self, client):
        """Test asking a question without any documents ingested"""
        response = await client.post("/ask", json={
            "question": "What is machine learning?",
            "session_id": "test_session_1"
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        # Should indicate no documents available
        assert len(data["sources"]) == 0

    @pytest.mark.asyncio
    async def test_clear_session(self, client):
        """Test clearing a session"""
        response = await client.post("/clear", json={
            "session_id": "test_session_1"
        })

        assert response.status_code == 200
        data = response.json()
        assert "cleared successfully" in data["message"]

    @pytest.mark.asyncio
    async def test_session_history(self, client):
        """Test getting session history"""
        # First ask a question
        await client.post("/ask", json={
            "question": "Test question",
            "session_id": "test_session_2"
        })

        # Then get history
        response = await client.get("/sessions/test_session_2/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert data["session_id"] == "test_session_2"

    @pytest.mark.asyncio
    async def test_upload_and_query_workflow(self, client):
        """Test the complete workflow: upload PDF, ask question"""
        # This test requires a sample PDF file
        sample_pdf_path = Path(__file__).parent / "sample.pdf"

        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not found for testing")

        # Upload PDF
        with open(sample_pdf_path, "rb") as f:
            files = {"file": ("sample.pdf", f, "application/pdf")}
            response = await client.post("/ingest", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "Successfully ingested" in data["message"]
        assert data["chunks_created"] > 0

        # Ask a question about the PDF
        response = await client.post("/ask", json={
            "question": "What is this document about?",
            "session_id": "test_session_3"
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        # Should have sources from the uploaded PDF
        assert len(data["sources"]) > 0


async def run_manual_test():
    """Manual test function to run without pytest"""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        print("Testing health check...")
        response = await client.get("/health")
        print(f"Health check: {response.status_code} - {response.json()}")

        print("\nTesting question without documents...")
        response = await client.post("/ask", json={
            "question": "What is artificial intelligence?",
            "session_id": "manual_test"
        })
        print(f"Ask without docs: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer'][:100]}...")
            print(f"Sources: {len(data['sources'])}")

        print("\nTesting session clear...")
        response = await client.post("/clear", json={
            "session_id": "manual_test"
        })
        print(f"Clear session: {response.status_code} - {response.json()}")


if __name__ == "__main__":
    # Run manual test if executed directly
    asyncio.run(run_manual_test())
import asyncio
import os
from pathlib import Path
from typing import List
from loguru import logger

from app.rag.ingestor import PDFIngestor
from app.config import settings


class StartupIngestion:
    def __init__(self, papers_dir: str = "./papers"):
        self.papers_dir = Path(papers_dir)
        self.ingestor = PDFIngestor()

    async def ingest_all_pdfs(self) -> List[dict]:
        """Ingest all PDF files from the papers directory"""

        if not self.papers_dir.exists():
            logger.warning(f"Papers directory {self.papers_dir} does not exist")
            return []

        # Find all PDF files
        pdf_files = list(self.papers_dir.glob("*.pdf"))

        if not pdf_files:
            logger.info(f"No PDF files found in {self.papers_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files for ingestion")

        results = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Ingesting {pdf_file.name}...")

                # Read PDF content
                with open(pdf_file, "rb") as f:
                    content = f.read()

                # Ingest the PDF
                result = await self.ingestor.ingest_pdf_content(content, pdf_file.name)

                logger.success(
                    f"✅ Successfully ingested {pdf_file.name}: "
                    f"{result.get('chunks_created', 0)} chunks created"
                )

                results.append({
                    "filename": pdf_file.name,
                    "status": "success",
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_created", 0),
                    "total_pages": result.get("total_pages", 0)
                })

            except Exception as e:
                logger.error(f"❌ Failed to ingest {pdf_file.name}: {str(e)}")
                results.append({
                    "filename": pdf_file.name,
                    "status": "error",
                    "error": str(e)
                })

        # Summary
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "error"])
        total_chunks = sum(r.get("chunks_created", 0) for r in results)

        logger.info(
            f"📊 Ingestion Summary: {successful}/{len(pdf_files)} files successful, "
            f"{total_chunks} total chunks created"
        )

        if failed > 0:
            logger.warning(f"⚠️  {failed} files failed to ingest")

        return results

    async def check_and_ingest(self) -> List[dict]:
        """Check if ingestion should run and perform it"""

        # Auto-ingestion can run in any mode
        try:
            return await self.ingest_all_pdfs()
        except Exception as e:
            logger.error(f"Startup ingestion failed: {str(e)}")
            return []


# Global instance
startup_ingestion = StartupIngestion()


async def run_startup_ingestion():
    """Main function to run startup ingestion"""
    logger.info("🚀 Starting automatic PDF ingestion...")
    results = await startup_ingestion.check_and_ingest()

    if results:
        logger.info("📚 PDF ingestion completed - ready to answer questions!")
    else:
        logger.info("📝 No PDFs ingested - system ready for manual uploads")

    return results
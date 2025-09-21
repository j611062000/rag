#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.ingestor import PDFIngestor
from app.config import settings
from loguru import logger


async def ingest_directory(directory_path: str):
    """Ingest all PDF files from a directory"""
    ingestor = PDFIngestor()
    pdf_files = list(Path(directory_path).glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to ingest")

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing {pdf_file.name}...")
            result = await ingestor.ingest_pdf_file(str(pdf_file))
            logger.success(
                f"Successfully ingested {pdf_file.name}: "
                f"{result['chunks_created']} chunks created"
            )
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_file.name}: {str(e)}")


async def ingest_single_file(file_path: str):
    """Ingest a single PDF file"""
    ingestor = PDFIngestor()

    try:
        logger.info(f"Processing {file_path}...")
        result = await ingestor.ingest_pdf_file(file_path)
        logger.success(
            f"Successfully ingested {file_path}: "
            f"{result['chunks_created']} chunks created"
        )
    except Exception as e:
        logger.error(f"Failed to ingest {file_path}: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdfs.py <file_or_directory_path>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    logger.info("Starting PDF ingestion...")
    logger.info(f"Vector DB: {settings.vector_db}")

    if os.path.isfile(path):
        if not path.endswith('.pdf'):
            logger.error("File must be a PDF")
            sys.exit(1)
        await ingest_single_file(path)
    elif os.path.isdir(path):
        await ingest_directory(path)
    else:
        logger.error(f"Invalid path: {path}")
        sys.exit(1)

    logger.info("PDF ingestion completed!")


if __name__ == "__main__":
    asyncio.run(main())
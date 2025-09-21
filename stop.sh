#!/bin/bash

echo "🛑 Stopping Chat-with-PDF RAG System..."

cd docker
docker-compose down

echo "✅ All services stopped successfully!"

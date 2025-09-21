#!/bin/bash

# 🚀 Chat-with-PDF RAG System Launcher
# Properly starts the system with OS environment variables (no hard-coded secrets)

echo "🔐 Chat-with-PDF RAG System"
echo "============================="

# Check if API key is provided as environment variable
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY environment variable is not set"
    echo ""
    echo "Usage:"
    echo "  export ANTHROPIC_API_KEY=\"your-api-key-here\""
    echo "  ./run_with_api_key.sh"
    echo ""
    echo "Or run directly:"
    echo "  ANTHROPIC_API_KEY=\"your-key\" ./run_with_api_key.sh"
    exit 1
fi

echo "✅ ANTHROPIC_API_KEY detected (${#ANTHROPIC_API_KEY} characters)"
echo ""

# Navigate to docker directory and start services
cd docker

echo "🚀 Starting Chat-with-PDF services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 15

echo ""
echo "🏥 Checking service health..."
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ API service is healthy" || echo "❌ API service not responding"

echo ""
echo "📋 Service URLs:"
echo "  🎨 Web UI: http://localhost:3000 (Start here!)"
echo "  🚀 API: http://localhost:8000"
echo "  🏥 Health: http://localhost:8000/health"
echo "  📚 Docs: http://localhost:8000/docs"

echo ""
echo "🧪 Test the system:"
echo '  curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '\''{"question": "What are the main findings in the research papers?"}'\'''

echo ""
echo "⏹️  To stop:"
echo "  cd docker && docker-compose down"
#!/bin/bash

# Chat-with-PDF RAG System - Quick Start Script
# This script will help you set up and run the entire RAG system with monitoring

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}  🤖 Chat-with-PDF RAG System Quick Start${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check requirements
check_requirements() {
    print_status "Checking system requirements..."

    local missing_tools=()

    if ! command_exists docker; then
        missing_tools+=("docker")
    fi

    if ! command_exists docker-compose; then
        missing_tools+=("docker-compose")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo "Please install:"
        echo "- Docker: https://docs.docker.com/get-docker/"
        echo "- Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi

    print_success "All required tools are installed"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking if Docker is running..."

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi

    print_success "Docker is running"
}

# Function to setup API keys
setup_api_keys() {
    print_status "Setting up API keys..."

    echo -e "\n${YELLOW}API Key Configuration:${NC}"
    echo "The system can work with multiple LLM providers:"
    echo "1. Anthropic Claude (Recommended) - Required for full functionality"
    echo "2. OpenAI GPT - Optional alternative"
    echo "3. Tavily Web Search - Optional for web search features"
    echo ""

    # Anthropic API Key (Required)
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${YELLOW}Anthropic Claude API Key (Required):${NC}"
        echo "Get your API key from: https://console.anthropic.com/"
        read -p "Enter your Anthropic API key (or press Enter to skip): " anthropic_key

        if [ ! -z "$anthropic_key" ]; then
            export ANTHROPIC_API_KEY="$anthropic_key"
            print_success "Anthropic API key configured"
        else
            print_warning "No Anthropic API key provided. System will run in mock mode."
        fi
    else
        print_success "Using existing Anthropic API key from environment"
    fi

    # OpenAI API Key (Optional)
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "\n${YELLOW}OpenAI API Key (Optional):${NC}"
        read -p "Enter your OpenAI API key (or press Enter to skip): " openai_key

        if [ ! -z "$openai_key" ]; then
            export OPENAI_API_KEY="$openai_key"
            print_success "OpenAI API key configured"
        fi
    else
        print_success "Using existing OpenAI API key from environment"
    fi

    # Tavily API Key (Optional)
    if [ -z "$TAVILY_API_KEY" ]; then
        echo -e "\n${YELLOW}Tavily Web Search API Key (Optional):${NC}"
        echo "Get your API key from: https://tavily.com/"
        read -p "Enter your Tavily API key (or press Enter to skip): " tavily_key

        if [ ! -z "$tavily_key" ]; then
            export TAVILY_API_KEY="$tavily_key"
            print_success "Tavily API key configured"
        fi
    else
        print_success "Using existing Tavily API key from environment"
    fi
}

# Function to setup PDF documents
setup_pdfs() {
    print_status "Checking PDF documents..."

    if [ ! -d "papers" ]; then
        mkdir -p papers
    fi

    pdf_count=$(find papers -name "*.pdf" 2>/dev/null | wc -l)

    if [ "$pdf_count" -eq 0 ]; then
        print_warning "No PDF documents found in ./papers/ directory"
        echo "The system will start without documents. You can:"
        echo "1. Add PDF files to the ./papers/ directory"
        echo "2. Restart the system to auto-ingest them"
        echo ""
        read -p "Continue without PDFs? (y/N): " continue_without_pdfs

        if [[ ! "$continue_without_pdfs" =~ ^[Yy]$ ]]; then
            echo "Please add PDF files to ./papers/ directory and run this script again."
            exit 0
        fi
    else
        print_success "Found $pdf_count PDF documents in ./papers/ directory"
    fi
}

# Function to build and start services
start_services() {
    print_status "Building and starting all services..."

    cd docker

    # Stop any existing services
    print_status "Stopping any existing services..."
    docker-compose down >/dev/null 2>&1 || true

    # Build and start services
    print_status "Building Docker images (this may take a few minutes)..."

    if [ ! -z "$ANTHROPIC_API_KEY" ]; then
        ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" docker-compose up --build -d
    else
        docker-compose up --build -d
    fi

    cd ..
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."

    local max_attempts=300
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            break
        fi

        print_status "Attempt $attempt/$max_attempts - waiting for API..."
        sleep 2
        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        print_error "Services failed to start within expected time"
        print_status "Checking service logs..."
        cd docker && docker-compose logs --tail=20
        exit 1
    fi

    print_success "All services are ready!"
}

# Function to display service information
show_service_info() {
    print_success "🎉 Chat-with-PDF RAG System is now running!"

    echo -e "\n${GREEN}📱 Available Services:${NC}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ 🌐 Web UI:          http://localhost:3000              │"
    echo "│ 🔌 API:             http://localhost:8000              │"
    echo "│ 📊 Grafana:         http://localhost:3001              │"
    echo "│ 🔍 Prometheus:      http://localhost:9090              │"
    echo "│ 🗄️  ChromaDB:       http://localhost:8001              │"
    echo "└─────────────────────────────────────────────────────────┘"

    echo -e "\n${GREEN}🔐 Grafana Login:${NC}"
    echo "Username: admin"
    echo "Password: admin123"

    echo -e "\n${GREEN}🚀 Quick Start:${NC}"
    echo "1. Open http://localhost:3000 in your browser"
    echo "2. Start asking questions about your PDF documents!"
    echo "3. Monitor system performance at http://localhost:3001"

    if [ ! -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "\n${GREEN}✅ Real AI Mode:${NC} Using Anthropic Claude API"
    else
        echo -e "\n${YELLOW}⚠️  Mock Mode:${NC} No API key provided - using mock responses"
        echo "   To use real AI, stop the system and set ANTHROPIC_API_KEY"
    fi

    echo -e "\n${BLUE}📋 Useful Commands:${NC}"
    echo "• Stop system:    ./stop.sh (or docker-compose down in ./docker/)"
    echo "• View logs:      cd docker && docker-compose logs -f"
    echo "• Restart:        ./start.sh"
    echo "• Run tests:      ./scripts/run_tests.sh"
}

# Function to create stop script
create_stop_script() {
    cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping Chat-with-PDF RAG System..."

cd docker
docker-compose down

echo "✅ All services stopped successfully!"
EOF

    chmod +x stop.sh
    print_success "Created stop.sh script for easy shutdown"
}

# Main execution
main() {
    print_header

    check_requirements
    check_docker
    setup_api_keys
    setup_pdfs
    start_services
    wait_for_services
    create_stop_script
    show_service_info

    echo -e "\n${GREEN}🎊 Setup completed successfully!${NC}"
    echo -e "Happy chatting with your PDFs! 📚🤖\n"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n${YELLOW}Setup interrupted by user${NC}"; exit 1' INT

# Run main function
main "$@"
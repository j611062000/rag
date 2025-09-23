#!/bin/bash

# 🧪 Chat-with-PDF Real-World Scenario Test Runner
# This script runs comprehensive tests after service startup

set -e

echo "🧪 Chat-with-PDF Real-World Scenario Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
TIMEOUT=240

# Function to check if service is ready
wait_for_service() {
    echo "⏳ Waiting for Chat-with-PDF service to be ready..."

    local count=0
    while [ $count -lt $TIMEOUT ]; do
        if curl -s "$API_URL/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Service is ready!${NC}"
            return 0
        fi

        echo "   Attempt $((count + 1))/$TIMEOUT - Service not ready yet..."
        sleep 2
        count=$((count + 1))
    done

    echo -e "${RED}❌ Service failed to start within $TIMEOUT seconds${NC}"
    return 1
}

# Function to start services if not running
start_services() {
    echo "🚀 Checking if services are running..."

    if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
        echo "📦 Starting Docker services..."
        cd docker
        docker-compose up -d
        cd ..

        # Wait for services to be ready
        if ! wait_for_service; then
            echo -e "${RED}Failed to start services${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Services are already running${NC}"
    fi
}

# Function to install Python dependencies if needed
install_dependencies() {
    echo "📦 Checking Python dependencies..."

    if ! python3 -c "import aiohttp" 2>/dev/null; then
        echo "Installing required Python packages..."
        pip3 install aiohttp asyncio || {
            echo -e "${YELLOW}⚠️  Could not install dependencies. Make sure you have pip3 installed.${NC}"
            echo "   You can install manually: pip3 install aiohttp"
        }
    fi
}

# Function to run the tests
run_tests() {
    echo ""
    echo "🧪 Running Real-World Scenario Tests..."
    echo "======================================"

    if [ -f "test_scenarios.py" ]; then
        python3 test_scenarios.py

        # Check if test results file was created
        if [ -f "test_results.json" ]; then
            echo ""
            echo -e "${BLUE}📄 Test results saved to: test_results.json${NC}"
        fi
    else
        echo -e "${RED}❌ Test script not found: test_scenarios.py${NC}"
        exit 1
    fi
}

# Function to show service logs
show_logs() {
    if [ "$1" = "--logs" ] || [ "$1" = "-l" ]; then
        echo ""
        echo "📋 Recent service logs:"
        echo "======================="
        cd docker
        docker-compose logs api --tail=20
        cd ..
    fi
}

# Function to cleanup on exit
cleanup() {
    if [ "$STOP_SERVICES" = "true" ]; then
        echo ""
        echo "🛑 Stopping services..."
        cd docker
        docker-compose down
        cd ..
    fi
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --logs     Show recent service logs after tests"
    echo "  -s, --stop     Stop services after tests"
    echo "  -q, --quick    Skip service startup check (assume running)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run tests with service startup"
    echo "  $0 --logs             # Run tests and show logs"
    echo "  $0 --stop             # Run tests and stop services"
    echo "  $0 --quick --logs     # Quick test with logs (assume services running)"
}

# Parse command line arguments
STOP_SERVICES="false"
QUICK_MODE="false"
SHOW_LOGS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--logs)
            SHOW_LOGS="true"
            shift
            ;;
        -s|--stop)
            STOP_SERVICES="true"
            trap cleanup EXIT
            shift
            ;;
        -q|--quick)
            QUICK_MODE="true"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo ""

    # Install dependencies
    install_dependencies

    # Start services unless in quick mode
    if [ "$QUICK_MODE" != "true" ]; then
        start_services
    else
        echo "⚡ Quick mode: Assuming services are already running"
        if ! wait_for_service; then
            echo -e "${RED}❌ Services not available in quick mode${NC}"
            exit 1
        fi
    fi

    # Run the tests
    run_tests

    # Show logs if requested
    if [ "$SHOW_LOGS" = "true" ]; then
        show_logs
    fi

    echo ""
    echo -e "${GREEN}🎉 Test execution completed!${NC}"

    # Additional information
    echo ""
    echo "📊 Quick Summary:"
    echo "• Check test_results.json for detailed results"
    echo "• Use 'docker-compose logs api' for service logs"
    echo "• Test individual endpoints: curl http://localhost:8000/ask"

    if [ "$STOP_SERVICES" != "true" ]; then
        echo ""
        echo -e "${YELLOW}💡 Services are still running at http://localhost:8000${NC}"
        echo "   Stop with: cd docker && docker-compose down"
    fi
}

# Run main function
main
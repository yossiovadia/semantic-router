#!/bin/bash
# start-mock-servers.sh - Start mock vLLM servers for testing
#
# This script starts mock vLLM servers that simulate multiple models
# for testing router classification functionality
#
# Signed-off-by: Yossi Ovadia <yovadia@redhat.com>

set -e

# Configuration
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$E2E_DIR/logs"
PIDS_FILE="$E2E_DIR/mock_pids.txt"

# Model configurations for mock servers
# Note: Primary model on 8000 matches router config, 8001 for multi-model testing
declare -A MOCK_MODELS=(
    ["8000"]="Qwen/Qwen2-0.5B-Instruct"
    ["8001"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


# Function to start servers in foreground for development
start_servers_foreground() {
    echo "Starting mock vLLM servers in FOREGROUND mode..."
    echo "==============================================="
    echo "Press Ctrl+C to stop all servers"
    echo "==============================================="

    # Create logs directory
    mkdir -p "$LOGS_DIR"

    # Check if ports are available
    for port in "${!MOCK_MODELS[@]}"; do
        if ! check_port "$port"; then
            echo "Error: Port $port is already in use. Please stop existing services."
            exit 1
        fi
    done

    # Array to store background process PIDs
    declare -a PIDS=()

    # Start servers in background but show output
    for port in "${!MOCK_MODELS[@]}"; do
        model="${MOCK_MODELS[$port]}"
        echo "🚀 Starting mock vLLM server on port $port with model $model..."

        # Start server and capture PID
        python3 "$E2E_DIR/mock-vllm-server.py" --port "$port" --model "$model" &
        local pid=$!
        PIDS+=($pid)

        echo "   ✅ Server started on port $port (PID: $pid)"
    done

    echo ""
    echo "🤖 Mock vLLM servers are running!"
    echo "Server endpoints:"
    for port in "${!MOCK_MODELS[@]}"; do
        model="${MOCK_MODELS[$port]}"
        echo "  📡 http://127.0.0.1:$port (model: $model)"
    done
    echo ""
    echo "🔍 You'll see request logs below as they come in..."
    echo "🛑 Press Ctrl+C to stop all servers"
    echo "$(printf '=%.0s' {1..50})"
    echo ""

    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "🛑 Stopping all mock servers..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Stopping PID $pid..."
                kill "$pid" 2>/dev/null || true
            fi
        done
        echo "✅ All mock servers stopped"
        exit 0
    }

    # Set up signal handlers
    trap cleanup SIGINT SIGTERM

    # Wait for all background processes
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
}


# Function to check if port is already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $port is already in use"
        return 1
    fi
    return 0
}


# Main execution - always run in foreground mode
start_servers_foreground
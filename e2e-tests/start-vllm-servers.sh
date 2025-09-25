#!/bin/bash
# start-vllm-servers.sh - Start multiple vLLM instances for semantic router testing
#
# This script starts vLLM servers on different ports to replace Ollama endpoints.
# Each server runs a different model to support the semantic router's model selection.

set -e

# Configuration
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$E2E_DIR")"
VLLM_LOG_DIR="$E2E_DIR/logs"
PIDS_FILE="$E2E_DIR/vllm_pids.txt"

# Models and ports configuration (single working model)
declare -A MODELS_CONFIG=(
    ["8000"]="Qwen/Qwen2-0.5B-Instruct"           # Working model on port 8000
)

# GPU memory settings (single model)
declare -A GPU_MEMORY_UTIL=(
    ["8000"]="0.3"  # Qwen2-0.5B with moderate memory usage
)

# Create log directory
mkdir -p "$VLLM_LOG_DIR"

# Function to check if vLLM is available
check_vllm_available() {
    if command -v vllm >/dev/null 2>&1; then
        echo "Found vLLM in PATH: $(command -v vllm)"
        return 0
    elif python -c "import vllm" >/dev/null 2>&1; then
        echo "Found vLLM via Python import"
        return 0
    else
        echo "Error: vLLM not found in current environment"
        echo ""
        echo "vLLM must be available to start the test servers."
        echo ""
        echo "If you have vLLM installed:"
        echo "  1. Activate your virtual environment that has vLLM:"
        echo "     source /path/to/your/vllm/venv/bin/activate"
        echo ""
        echo "  2. Or ensure vLLM is in your PATH if installed globally"
        echo ""
        echo "If you need to install vLLM:"
        echo "  1. Create and activate a virtual environment:"
        echo "     python -m venv vllm-env"
        echo "     source vllm-env/bin/activate"
        echo ""
        echo "  2. Install vLLM:"
        echo "     pip install vllm"
        echo ""
        echo "  3. Re-run this script with the environment activated"
        echo ""
        exit 1
    fi
}

# Function to check if port is available
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        echo "Warning: Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to start a vLLM server
start_vllm_server() {
    local port=$1
    local model=$2
    local log_file="$VLLM_LOG_DIR/vllm_${port}.log"
    
    echo "Starting vLLM server on port $port with model $model..."
    
    # Check if port is available
    if ! check_port "$port"; then
        echo "Skipping port $port (already in use)"
        return 1
    fi
    
    # Get GPU memory utilization for this port
    local gpu_mem_util="${GPU_MEMORY_UTIL[$port]}"
    
    # Start vLLM server in background
    # Use current environment (assumes vLLM is available)
    vllm serve "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --gpu-memory-utilization "$gpu_mem_util" \
        --max-model-len 2048 \
        --disable-log-requests \
        --trust-remote-code \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "$pid" >> "$PIDS_FILE"
    echo "Started vLLM server (PID: $pid) on port $port with model $model"
    echo "Log file: $log_file"
    
    return 0
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for server on port $port to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "Server on port $port is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 2
        echo "Attempt $attempt/$max_attempts..."
    done
    
    echo "Error: Server on port $port failed to start or is not responding"
    return 1
}

# Main execution
main() {
    echo "Starting vLLM servers for semantic router..."
    
    # Check prerequisites
    check_vllm_available
    
    # Clean up any existing PID file
    rm -f "$PIDS_FILE"
    
    # Start each configured server
    local started_servers=0
    for port in "${!MODELS_CONFIG[@]}"; do
        model="${MODELS_CONFIG[$port]}"
        if start_vllm_server "$port" "$model"; then
            started_servers=$((started_servers + 1))
        fi
    done
    
    if [ $started_servers -eq 0 ]; then
        echo "Error: No vLLM servers were started"
        exit 1
    fi
    
    echo ""
    echo "Started $started_servers vLLM server(s). Waiting for them to be ready..."
    echo ""
    
    # Wait for all servers to be ready
    local ready_servers=0
    for port in "${!MODELS_CONFIG[@]}"; do
        if wait_for_server "$port"; then
            ready_servers=$((ready_servers + 1))
        fi
    done
    
    echo ""
    echo "=== vLLM Servers Status ==="
    echo "Started: $started_servers"
    echo "Ready: $ready_servers"
    echo ""
    
    if [ $ready_servers -gt 0 ]; then
        echo "vLLM servers are ready for semantic router testing!"
        echo ""
        echo "Available endpoints:"
        for port in "${!MODELS_CONFIG[@]}"; do
            model="${MODELS_CONFIG[$port]}"
            echo "  - http://127.0.0.1:$port (Model: $model)"
        done
        echo ""
        echo "Health check: curl http://127.0.0.1:8000/health"
        echo "Models list: curl http://127.0.0.1:8000/v1/models"
        echo ""
        echo "To stop servers, run: ./e2e-tests/stop-vllm-servers.sh"
    else
        echo "Error: No servers are ready. Check logs in $VLLM_LOG_DIR/"
        exit 1
    fi
}

# Run main function
main "$@"
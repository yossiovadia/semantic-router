#!/usr/bin/env bash
# =============================================================================
# vSR Egress Routing Demo - Launcher
#
# Starts all required services:
#   1. Mock servers (internal model + Anthropic mock)
#   2. vSR router (ExtProc on port 50051)
#   3. Envoy proxy (port 8801)
#
# Prerequisites:
#   - vSR router built: make build-router
#   - ML models downloaded: make download-models
#   - func-e (Envoy runner) installed
#   - OPENAI_API_KEY set for real OpenAI calls
#
# Usage:
#   ./run-egress-demo.sh           # Start all services
#   ./run-egress-demo.sh --stop    # Stop all services
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PID_DIR="/tmp/vsr-egress-demo"
MOCK_PID_FILE="$PID_DIR/mock-server.pid"
ROUTER_PID_FILE="$PID_DIR/router.pid"
ENVOY_PID_FILE="$PID_DIR/envoy.pid"

MOCK_LOG="/tmp/vsr-demo-mock.log"
ROUTER_LOG="/tmp/vsr-demo-router.log"
ENVOY_LOG="/tmp/vsr-demo-envoy.log"

stop_services() {
    echo "Stopping vSR egress demo services..."
    for pidfile in "$ENVOY_PID_FILE" "$ROUTER_PID_FILE" "$MOCK_PID_FILE"; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            kill "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    # Clean up any stray processes
    pkill -f "mock-server.py.*8002" 2>/dev/null || true
    pkill -f "router.*config.egress-demo" 2>/dev/null || true
    lsof -ti:8801 2>/dev/null | xargs kill -9 2>/dev/null || true
    echo "All services stopped."
}

if [ "${1:-}" = "--stop" ]; then
    stop_services
    exit 0
fi

# ─── Pre-flight checks ───
echo "=============================================="
echo "  vSR Egress Routing Demo"
echo "=============================================="
echo ""

if [ ! -f "$PROJECT_ROOT/bin/router" ]; then
    echo "ERROR: Router binary not found. Run 'make build-router' first."
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set. OpenAI scenarios will fail."
    echo "         Set it with: export OPENAI_API_KEY=sk-proj-..."
    echo ""
fi

# ─── Clean up old processes ───
echo "0. Cleaning up any existing services..."
stop_services
sleep 1
mkdir -p "$PID_DIR"
rm -f "$MOCK_LOG" "$ROUTER_LOG" "$ENVOY_LOG"

# ─── Start mock servers ───
echo ""
echo "1. Starting mock servers (internal model on 8002, Anthropic mock on 8003)..."
nohup python3 "$SCRIPT_DIR/mock-server.py" --internal-port 8002 --anthropic-port 8003 \
    > "$MOCK_LOG" 2>&1 & echo $! > "$MOCK_PID_FILE"
sleep 2

curl -sf http://127.0.0.1:8002/health > /dev/null 2>&1 && \
    echo "   OK: Mock internal model is healthy" || \
    echo "   WARN: Mock internal model may not be ready"
curl -sf http://127.0.0.1:8003/health > /dev/null 2>&1 && \
    echo "   OK: Mock Anthropic is healthy" || \
    echo "   WARN: Mock Anthropic may not be ready"

# ─── Start vSR router ───
echo ""
echo "2. Starting vSR router (ExtProc on port 50051)..."
export LD_LIBRARY_PATH="${PROJECT_ROOT}/candle-binding/target/release:${PROJECT_ROOT}/ml-binding/target/release:${LD_LIBRARY_PATH:-}"
nohup "$PROJECT_ROOT/bin/router" -config="$PROJECT_ROOT/config/demo/config.egress-demo.yaml" \
    > "$ROUTER_LOG" 2>&1 & echo $! > "$ROUTER_PID_FILE"
echo "   Waiting for router to initialize models (15s)..."
sleep 15
echo "   OK: Router started (check $ROUTER_LOG for details)"

# ─── Start Envoy ───
echo ""
echo "3. Starting Envoy proxy (port 8801)..."
if ! command -v func-e >/dev/null 2>&1; then
    echo "   func-e not found. Installing..."
    curl -sL https://func-e.io/install.sh | sudo bash -s -- -b /usr/local/bin
fi
nohup func-e run --config-path "$PROJECT_ROOT/config/demo/envoy.egress-demo.yaml" \
    > "$ENVOY_LOG" 2>&1 & echo $! > "$ENVOY_PID_FILE"
sleep 3
echo "   OK: Envoy started"

# ─── Ready ───
echo ""
echo "=============================================="
echo "  Demo Ready"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  Gateway:        http://localhost:8801"
echo "  Router (gRPC):  localhost:50051"
echo "  Mock Internal:  http://localhost:8002"
echo "  Mock Anthropic: http://localhost:8003"
echo "  Envoy Admin:    http://localhost:19000"
echo ""
echo "Run demo scenarios:"
echo "  $SCRIPT_DIR/run-scenarios.sh              # MVP demo"
echo "  $SCRIPT_DIR/run-scenarios.sh --all        # All scenarios"
echo "  $SCRIPT_DIR/run-scenarios.sh --security   # Security bonus"
echo ""
echo "Logs:"
echo "  Router: tail -f $ROUTER_LOG"
echo "  Envoy:  tail -f $ENVOY_LOG"
echo "  Mocks:  tail -f $MOCK_LOG"
echo ""
echo "Stop:"
echo "  $0 --stop"
echo ""
echo "Press Enter to stop all services..."
read -r _
stop_services

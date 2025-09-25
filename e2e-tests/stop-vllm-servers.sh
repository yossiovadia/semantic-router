#!/bin/bash
# stop-vllm-servers.sh - Stop all running vLLM servers
#
# This script stops all vLLM servers started by start-vllm-servers.sh

set -e

# Configuration
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$E2E_DIR/vllm_pids.txt"

# Function to stop servers by PID file
stop_by_pids() {
    if [ ! -f "$PIDS_FILE" ]; then
        echo "No PID file found at $PIDS_FILE"
        return 1
    fi
    
    local stopped_count=0
    
    while IFS= read -r pid; do
        if [ -n "$pid" ] && [ "$pid" != "0" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping vLLM server (PID: $pid)..."
                if kill -TERM "$pid" 2>/dev/null; then
                    # Wait up to 10 seconds for graceful shutdown
                    local wait_count=0
                    while [ $wait_count -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
                        sleep 1
                        wait_count=$((wait_count + 1))
                    done
                    
                    # Force kill if still running
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "Force killing PID $pid..."
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                    
                    stopped_count=$((stopped_count + 1))
                    echo "Stopped PID $pid"
                else
                    echo "Failed to stop PID $pid"
                fi
            else
                echo "PID $pid is not running"
            fi
        fi
    done < "$PIDS_FILE"
    
    # Clean up PID file
    rm -f "$PIDS_FILE"
    
    echo "Stopped $stopped_count vLLM server(s)"
    return 0
}

# Function to stop servers by process name (fallback)
stop_by_process_name() {
    echo "Searching for vLLM processes..."
    
    # Find vLLM serve processes
    local pids=$(pgrep -f "vllm serve" || true)
    
    if [ -z "$pids" ]; then
        echo "No vLLM serve processes found"
        return 1
    fi
    
    local stopped_count=0
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping vLLM process (PID: $pid)..."
            if kill -TERM "$pid" 2>/dev/null; then
                # Wait up to 10 seconds for graceful shutdown
                local wait_count=0
                while [ $wait_count -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
                    sleep 1
                    wait_count=$((wait_count + 1))
                done
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing PID $pid..."
                    kill -KILL "$pid" 2>/dev/null || true
                fi
                
                stopped_count=$((stopped_count + 1))
                echo "Stopped PID $pid"
            else
                echo "Failed to stop PID $pid"
            fi
        fi
    done
    
    echo "Stopped $stopped_count vLLM process(es)"
    return 0
}

# Function to check if any vLLM servers are still running
check_remaining_processes() {
    local remaining=$(pgrep -f "vllm serve" | wc -l || echo "0")
    if [ "$remaining" -gt 0 ]; then
        echo "Warning: $remaining vLLM processes may still be running"
        echo "You can check with: ps aux | grep 'vllm serve'"
        echo "Check logs in: $E2E_DIR/logs/"
    else
        echo "All vLLM servers appear to be stopped"
    fi
}

# Main execution
main() {
    echo "Stopping vLLM servers..."
    
    # Try to stop by PID file first
    if ! stop_by_pids; then
        echo "Trying to stop by process name..."
        stop_by_process_name
    fi
    
    # Give processes time to fully terminate
    sleep 2
    
    # Check if any processes are still running
    check_remaining_processes
    
    echo "Done."
}

# Run main function
main "$@"
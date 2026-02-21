#!/bin/bash
# VSR for OpenClaw — vector store only (no routing, no vLLM, no Envoy)
# Usage: ./vsr-openclaw.sh

set -e

cd "$(dirname "$0")"

export DYLD_LIBRARY_PATH="${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release"

echo "Starting VSR (vector store only) for OpenClaw..."
echo "  API:    http://127.0.0.1:8080"
echo "  Config: config/openclaw-memory-only.yaml"
echo ""

exec ./bin/router \
  -config=config/openclaw-memory-only.yaml \
  -api-port=8080 \
  -port=50051 \
  -enable-api=true

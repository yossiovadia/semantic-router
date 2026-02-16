#!/bin/bash
# =============================================================================
# Deploy vSR Egress Routing Demo to OpenShift
#
# Deploys:
#   - llm-katan (mock external provider + mock internal model)
#   - Mock Anthropic endpoint
#   - vSR router (ExtProc with egress config)
#   - Envoy proxy
#   - Demo web UI
#
# Prerequisites:
#   - oc login to OpenShift cluster
#   - ghcr.io/vllm-project/semantic-router/extproc:latest accessible
#
# Usage:
#   ./deploy.sh                 # Deploy with echo backend (no GPU, instant)
#   ./deploy.sh --real-model    # Deploy with Qwen3-0.6B (CPU inference)
#   ./deploy.sh --cleanup       # Remove everything
# =============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="vsr-egress-demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USE_REAL_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --real-model) USE_REAL_MODEL=true; shift ;;
        --cleanup) cleanup=true; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --real-model    Use Qwen3-0.6B for external provider (CPU, slower)"
            echo "  --cleanup       Remove all demo resources"
            echo "  --namespace NS  Custom namespace (default: vsr-egress-demo)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()     { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Cleanup ───
if [[ "${cleanup:-}" == "true" ]]; then
    log "Cleaning up namespace: $NAMESPACE"
    oc delete namespace "$NAMESPACE" --ignore-not-found=true
    success "Cleanup complete"
    exit 0
fi

# ─── Pre-flight ───
if ! oc whoami &>/dev/null; then
    error "Not logged in to OpenShift. Run: oc login <server>"
fi
success "Logged in as $(oc whoami)"

# ─── Create namespace ───
log "Creating namespace: $NAMESPACE"
oc create namespace "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
success "Namespace ready"

# ─── Deploy mock Anthropic server ───
log "Deploying mock Anthropic endpoint..."
oc create configmap mock-anthropic-server \
    --from-file=mock-server.py="$PROJECT_ROOT/e2e/testing/demo/mock-server.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/mock-anthropic.yaml"

# ─── Deploy llm-katan (external + internal provider) ───
log "Deploying llm-katan instances..."
if [[ "$USE_REAL_MODEL" == "true" ]]; then
    log "Using real model (Qwen3-0.6B) — pods will need time to download model"
    oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/llm-katan-real.yaml"
else
    log "Using echo backend — instant responses, no model download"
    oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/llm-katan-echo.yaml"
fi

# ─── Deploy vSR router + Envoy ───
log "Creating ConfigMaps..."
oc create configmap vsr-egress-config \
    --from-file=config.yaml="$SCRIPT_DIR/config-egress-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc create configmap envoy-egress-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/envoy-egress-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

log "Deploying vSR router + Envoy..."
oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/vsr-deployment.yaml"

# ─── Deploy demo web UI ───
log "Deploying demo web UI..."
oc create configmap demo-ui-files \
    --from-file=demo.html="$PROJECT_ROOT/e2e/testing/demo/demo.html" \
    --from-file=demo-server.py="$PROJECT_ROOT/e2e/testing/demo/demo-server.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/demo-ui.yaml"

# ─── Create Routes ───
log "Creating OpenShift Routes..."

# Route for demo web UI (public)
oc expose service demo-ui --name=demo-ui-route -n "$NAMESPACE" \
    --dry-run=client -o yaml | oc apply -f -

# Route for gateway (for direct API access)
oc expose service vsr-gateway --name=gateway-route -n "$NAMESPACE" \
    --dry-run=client -o yaml | oc apply -f -

# ─── Wait for pods ───
log "Waiting for pods to be ready..."
oc wait --for=condition=Ready pod -l app=mock-anthropic -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "mock-anthropic not ready yet"
oc wait --for=condition=Ready pod -l app=llm-katan -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "llm-katan not ready yet"
oc wait --for=condition=Ready pod -l app=vsr-router -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "vsr-router not ready yet"
oc wait --for=condition=Ready pod -l app=demo-ui -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "demo-ui not ready yet"

# ─── Print URLs ───
DEMO_URL=$(oc get route demo-ui-route -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "pending")
GW_URL=$(oc get route gateway-route -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "pending")

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  vSR Egress Routing Demo Deployed${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Demo UI:  ${BLUE}http://${DEMO_URL}${NC}"
echo -e "  Gateway:  ${BLUE}http://${GW_URL}${NC}"
echo ""
echo -e "  Share the Demo UI URL with stakeholders."
echo ""
echo -e "  Cleanup:  $0 --cleanup"
echo ""

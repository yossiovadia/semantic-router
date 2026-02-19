#!/bin/bash
# =============================================================================
# Deploy vSR Egress Routing Demo to OpenShift
#
# Deploys:
#   Phase A/B: ExtProc + BBR demo with mock backends
#   Phase C:   Real auth (Kuadrant/Authorino + MaaS API), external provider
#              simulation (separate namespace)
#
# Prerequisites:
#   - oc login to OpenShift cluster (admin access for Kuadrant install)
#   - Container images accessible (quay.io/jabadia/*, ghcr.io/vllm-project/*)
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
EXT_NAMESPACE="external-providers"
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
    log "Cleaning up demo resources..."
    # Phase C resources (cluster-scoped)
    oc delete authpolicy vsr-demo-auth-policy -n openshift-ingress --ignore-not-found=true 2>/dev/null || true
    oc delete httproute vsr-gateway-route -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
    oc delete gateway vsr-demo-gateway -n openshift-ingress --ignore-not-found=true 2>/dev/null || true
    oc delete clusterrolebinding maas-api-vsr-demo --ignore-not-found=true 2>/dev/null || true
    oc delete clusterrole maas-api-vsr-demo --ignore-not-found=true 2>/dev/null || true
    # Namespaces
    oc delete namespace vsr-demo-tier-premium --ignore-not-found=true 2>/dev/null || true
    oc delete namespace "$EXT_NAMESPACE" --ignore-not-found=true 2>/dev/null || true
    oc delete namespace "$NAMESPACE" --ignore-not-found=true
    # Note: kuadrant-system left in place (reusable), GatewayClass left in place (shared)
    success "Cleanup complete"
    exit 0
fi

# ─── Pre-flight ───
if ! oc whoami &>/dev/null; then
    error "Not logged in to OpenShift. Run: oc login <server>"
fi
success "Logged in as $(oc whoami)"

# =============================================================================
# PHASE A/B: Core Demo Infrastructure
# =============================================================================

# ─── Create namespaces ───
log "Creating namespaces..."
oc create namespace "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
oc create namespace "$EXT_NAMESPACE" --dry-run=client -o yaml | oc apply -f -
# Allow external-providers to pull images built in vsr-egress-demo
oc policy add-role-to-group system:image-puller "system:serviceaccounts:${EXT_NAMESPACE}" -n "$NAMESPACE" 2>/dev/null || true
success "Namespaces ready"

# ─── Build images on cluster ───
log "Building mock-anthropic image on cluster..."
if ! oc get imagestream mock-anthropic -n "$NAMESPACE" &>/dev/null; then
    oc new-build --name mock-anthropic --binary --strategy=docker -n "$NAMESPACE"
fi
BUILD_TMP=$(mktemp -d)
cp "$SCRIPT_DIR/Dockerfile.mock-anthropic" "$BUILD_TMP/Dockerfile"
cp "$SCRIPT_DIR/mock-server.py" "$BUILD_TMP/"
oc start-build mock-anthropic --from-dir="$BUILD_TMP" --follow -n "$NAMESPACE" || true
rm -rf "$BUILD_TMP"
success "mock-anthropic image built"

log "Building demo-ui image on cluster..."
if ! oc get imagestream demo-ui -n "$NAMESPACE" &>/dev/null; then
    oc new-build --name demo-ui --binary --strategy=docker -n "$NAMESPACE"
fi
BUILD_TMP=$(mktemp -d)
cp "$SCRIPT_DIR/Dockerfile.demo-ui" "$BUILD_TMP/Dockerfile"
cp "$SCRIPT_DIR/demo.html" "$BUILD_TMP/"
cp "$SCRIPT_DIR/admin.html" "$BUILD_TMP/"
cp "$SCRIPT_DIR/demo-server.py" "$BUILD_TMP/"
oc start-build demo-ui --from-dir="$BUILD_TMP" --follow -n "$NAMESPACE" || true
rm -rf "$BUILD_TMP"
success "demo-ui image built"

# ─── Deploy external providers (separate namespace) ───
log "Deploying external providers (simulated external zone)..."
oc apply -f "$SCRIPT_DIR/phase-c/external-providers.yaml"

# ─── Deploy internal model only ───
log "Deploying internal model (llm-katan-internal)..."
if [[ "$USE_REAL_MODEL" == "true" ]]; then
    log "Using real model (Qwen3-0.6B) — pods will need time to download model"
    # Apply only the internal model from llm-katan-real.yaml
    oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/llm-katan-real.yaml"
else
    log "Using echo backend — instant responses, no model download"
    # Apply only internal model — external is in external-providers namespace
    oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/llm-katan-echo.yaml"
fi
# Delete any leftover external provider deployments from the main namespace
oc delete deployment llm-katan-external -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
oc delete service llm-katan-external -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
oc delete deployment mock-anthropic -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
oc delete service mock-anthropic -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true

# ─── Deploy vSR router + Envoy ───
log "Creating ConfigMaps..."
oc create configmap vsr-egress-config \
    --from-file=config.yaml="$SCRIPT_DIR/config-egress-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc create configmap envoy-egress-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/envoy-egress-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

log "Deploying vSR router + Envoy (ExtProc)..."
oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/vsr-deployment.yaml"

# ─── Deploy BBR router + Envoy (BBR Plugin) ───
log "Deploying BBR router + Envoy (BBR Plugin)..."
oc create configmap envoy-bbr-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/envoy-bbr-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/bbr-deployment.yaml"

# ─── Deploy demo web UI ───
log "Deploying demo web UI..."
oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/demo-ui.yaml"

# ─── Create Routes ───
log "Creating OpenShift Routes..."

# Route for demo web UI (HTTPS with edge TLS termination)
oc create route edge demo-ui-route --service=demo-ui --port=http \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

# Route for ExtProc gateway (HTTP — API access, direct bypass)
oc expose service vsr-gateway --name=gateway-route -n "$NAMESPACE" \
    --dry-run=client -o yaml | oc apply -f -

# Route for BBR gateway (HTTP — API access)
oc expose service bbr-gateway --name=bbr-gateway-route -n "$NAMESPACE" \
    --dry-run=client -o yaml | oc apply -f -

# =============================================================================
# PHASE C: Real Auth Infrastructure
# =============================================================================

log ""
log "═══════════════════════════════════════════════"
log "  Phase C: Installing Auth Infrastructure"
log "═══════════════════════════════════════════════"

# ─── Install Kuadrant operator ───
log "Installing Kuadrant operator v1.3.1..."
oc apply -f "$SCRIPT_DIR/phase-c/kuadrant-install.yaml"

# Wait for Kuadrant subscription to be installed
log "Waiting for Kuadrant operator subscription..."
KUADRANT_NS="kuadrant-system"
for i in $(seq 1 60); do
    CSV=$(oc get csv -n "$KUADRANT_NS" --no-headers 2>/dev/null | grep "^kuadrant-operator" | awk '{print $1}' | head -1)
    if [[ -n "$CSV" ]]; then
        PHASE=$(oc get csv "$CSV" -n "$KUADRANT_NS" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        if [[ "$PHASE" == "Succeeded" ]]; then
            success "Kuadrant operator installed: $CSV"
            break
        fi
    fi
    if [[ $i -eq 60 ]]; then
        warn "Kuadrant operator not ready after 300s — continuing anyway"
    fi
    sleep 5
done

# ─── Patch Kuadrant CSV for OpenShift Gateway controller ───
log "Patching Kuadrant CSV for OpenShift Gateway controller..."
CSV=$(oc get csv -n "$KUADRANT_NS" --no-headers 2>/dev/null | grep "^kuadrant-operator" | awk '{print $1}' | head -1)
if [[ -n "$CSV" ]]; then
    # Find the ISTIO_GATEWAY_CONTROLLER_NAMES env var index
    ENV_INDEX=$(oc get csv "$CSV" -n "$KUADRANT_NS" -o json 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); envs=d['spec']['install']['spec']['deployments'][0]['spec']['template']['spec']['containers'][0].get('env',[]); print(next((str(i) for i,e in enumerate(envs) if e['name']=='ISTIO_GATEWAY_CONTROLLER_NAMES'), 'NONE'))" 2>/dev/null || echo "NONE")

    if [[ "$ENV_INDEX" == "NONE" ]]; then
        # Add the env var
        oc patch csv "$CSV" -n "$KUADRANT_NS" --type='json' -p='[
          {"op": "add", "path": "/spec/install/spec/deployments/0/spec/template/spec/containers/0/env/-",
           "value": {"name": "ISTIO_GATEWAY_CONTROLLER_NAMES", "value": "istio.io/gateway-controller,openshift.io/gateway-controller/v1"}}
        ]' 2>/dev/null || warn "Failed to add ISTIO_GATEWAY_CONTROLLER_NAMES"
    else
        # Update existing env var
        oc patch csv "$CSV" -n "$KUADRANT_NS" --type='json' -p="[
          {\"op\": \"replace\", \"path\": \"/spec/install/spec/deployments/0/spec/template/spec/containers/0/env/${ENV_INDEX}/value\",
           \"value\": \"istio.io/gateway-controller,openshift.io/gateway-controller/v1\"}
        ]" 2>/dev/null || warn "Failed to update ISTIO_GATEWAY_CONTROLLER_NAMES"
    fi

    # Force restart operator pod to pick up new env
    log "Restarting Kuadrant operator to apply Gateway controller config..."
    oc delete pod -n "$KUADRANT_NS" -l control-plane=controller-manager --force --grace-period=0 2>/dev/null || \
        oc delete pod -n "$KUADRANT_NS" -l app.kubernetes.io/name=kuadrant-operator --force --grace-period=0 2>/dev/null || true
    sleep 10
    oc rollout status deployment/kuadrant-operator-controller-manager -n "$KUADRANT_NS" --timeout=120s 2>/dev/null || warn "Operator rollout timeout"

    # Verify env var
    VERIFY_ENV=$(oc exec -n "$KUADRANT_NS" deployment/kuadrant-operator-controller-manager -- env 2>/dev/null | grep ISTIO_GATEWAY_CONTROLLER_NAMES || echo "")
    if [[ "$VERIFY_ENV" == *"openshift.io/gateway-controller/v1"* ]]; then
        success "Kuadrant operator patched for OpenShift Gateway controller"
    else
        warn "Operator env may not have correct value: $VERIFY_ENV"
    fi

    # Wait for operator to fully initialize
    log "Waiting 15s for operator initialization..."
    sleep 15
else
    warn "Could not find Kuadrant CSV, skipping Gateway controller patch"
fi

# ─── Apply GatewayClass ───
log "Creating GatewayClass..."
oc apply -f "$SCRIPT_DIR/phase-c/gatewayclass.yaml"
success "GatewayClass ready"

# ─── Create Gateway ───
log "Creating Gateway..."
CLUSTER_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}' 2>/dev/null || echo "")
if [[ -z "$CLUSTER_DOMAIN" ]]; then
    error "Could not determine cluster domain"
fi
export CLUSTER_DOMAIN
envsubst '${CLUSTER_DOMAIN}' < "$SCRIPT_DIR/phase-c/gateway.yaml" | oc apply -f -
success "Gateway created (hostname: vsr-demo.${CLUSTER_DOMAIN})"

# Wait for Gateway to be Programmed
log "Waiting for Gateway to be Programmed..."
if oc wait --for=condition=Programmed gateway/vsr-demo-gateway -n openshift-ingress --timeout=120s 2>/dev/null; then
    success "Gateway is Programmed"
else
    warn "Gateway not yet Programmed — AuthPolicy may take longer"
fi

# ─── Apply Kuadrant CR ───
log "Activating Kuadrant (Authorino + Limitador)..."
oc apply -f "$SCRIPT_DIR/phase-c/kuadrant-cr.yaml"

# Wait for Kuadrant to be ready
log "Waiting for Kuadrant to become ready..."
for i in $(seq 1 36); do
    READY=$(oc get kuadrant kuadrant -n "$KUADRANT_NS" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    if [[ "$READY" == "True" ]]; then
        success "Kuadrant is ready"
        break
    fi
    if [[ $i -eq 36 ]]; then
        REASON=$(oc get kuadrant kuadrant -n "$KUADRANT_NS" -o jsonpath='{.status.conditions[?(@.type=="Ready")].reason}' 2>/dev/null || echo "unknown")
        if [[ "$REASON" == "MissingDependency" ]]; then
            log "Kuadrant MissingDependency — restarting operator..."
            oc delete pod -n "$KUADRANT_NS" -l control-plane=controller-manager --force --grace-period=0 2>/dev/null || true
            sleep 15
        fi
        warn "Kuadrant not ready (reason: $REASON) — continuing"
    fi
    sleep 5
done

# ─── Create KServe CRD (needed by MaaS API informers) ───
log "Creating LLMInferenceService CRD (stub for MaaS API)..."
if ! oc get crd llminferenceservices.serving.kserve.io &>/dev/null; then
    oc apply -f - <<'CRDEOF'
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: llminferenceservices.serving.kserve.io
spec:
  group: serving.kserve.io
  versions:
  - name: v1alpha1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        x-kubernetes-preserve-unknown-fields: true
  scope: Namespaced
  names:
    plural: llminferenceservices
    singular: llminferenceservice
    kind: LLMInferenceService
    shortNames: [llm]
CRDEOF
    success "CRD created"
else
    success "CRD already exists"
fi

# ─── Deploy MaaS API ───
log "Deploying MaaS API (tier lookup)..."
oc apply -f "$SCRIPT_DIR/phase-c/maas-api.yaml"
oc wait --for=condition=Ready pod -l app=maas-api -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "MaaS API not ready yet"

# Verify MaaS API health
MAAS_HEALTH=$(oc exec -n "$NAMESPACE" deployment/maas-api -- curl -sf localhost:8080/health 2>/dev/null || echo "FAIL")
if [[ "$MAAS_HEALTH" != "FAIL" ]]; then
    success "MaaS API healthy"
else
    warn "MaaS API health check failed"
fi

# ─── Create demo users ───
log "Creating demo users and tier groups..."
oc apply -f "$SCRIPT_DIR/phase-c/demo-users.yaml"
success "Demo users created (free-user, premium-user)"

# ─── Apply AuthPolicy ───
log "Applying AuthPolicy..."
oc apply -f "$SCRIPT_DIR/phase-c/auth-policy.yaml"

# Wait for AuthPolicy to be accepted
log "Waiting for AuthPolicy to be accepted..."
for i in $(seq 1 24); do
    ACCEPTED=$(oc get authpolicy vsr-demo-auth-policy -n openshift-ingress -o jsonpath='{.status.conditions[?(@.type=="Accepted")].status}' 2>/dev/null || echo "")
    if [[ "$ACCEPTED" == "True" ]]; then
        success "AuthPolicy accepted"
        break
    fi
    if [[ $i -eq 24 ]]; then
        warn "AuthPolicy not accepted yet — auth may not be enforced"
    fi
    sleep 5
done

# ─── Apply HTTPRoute ───
log "Applying HTTPRoute..."
oc apply -f "$SCRIPT_DIR/phase-c/httproute.yaml"
success "HTTPRoute created"

# ─── Generate demo tokens ───
log "Generating demo user tokens..."
FREE_TOKEN=$(oc create token free-user -n "$NAMESPACE" --audience vsr-demo-gateway-sa --duration=24h 2>/dev/null || echo "")
PREMIUM_TOKEN=$(oc create token premium-user -n vsr-demo-tier-premium --audience vsr-demo-gateway-sa --duration=24h 2>/dev/null || echo "")

if [[ -n "$FREE_TOKEN" && -n "$PREMIUM_TOKEN" ]]; then
    oc create secret generic demo-tokens \
        --from-literal=free-token="$FREE_TOKEN" \
        --from-literal=premium-token="$PREMIUM_TOKEN" \
        -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
    success "Demo tokens generated and stored in ConfigMap"
else
    warn "Failed to generate demo tokens"
fi

# ─── Store Gateway host for demo UI ───
AUTH_GW_HOST="vsr-demo.${CLUSTER_DOMAIN}"
oc create configmap demo-config \
    --from-literal=auth-gateway-host="$AUTH_GW_HOST" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

# ─── Wait for all pods ───
log "Waiting for all pods to be ready..."
oc wait --for=condition=Ready pod -l app=mock-anthropic -n "$EXT_NAMESPACE" --timeout=120s 2>/dev/null || warn "mock-anthropic not ready yet"
oc wait --for=condition=Ready pod -l app=llm-katan,role=external-provider -n "$EXT_NAMESPACE" --timeout=120s 2>/dev/null || warn "llm-katan-external not ready yet"
oc wait --for=condition=Ready pod -l app=llm-katan,role=internal-model -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "llm-katan-internal not ready yet"
oc wait --for=condition=Ready pod -l app=vsr-router -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "vsr-router not ready yet"
oc wait --for=condition=Ready pod -l app=bbr-router -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "bbr-router not ready yet"
oc wait --for=condition=Ready pod -l app=demo-ui -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "demo-ui not ready yet"

# ─── Print URLs ───
DEMO_URL=$(oc get route demo-ui-route -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "pending")
GW_URL=$(oc get route gateway-route -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "pending")
BBR_URL=$(oc get route bbr-gateway-route -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "pending")
AUTH_GW_URL="vsr-demo.${CLUSTER_DOMAIN}"

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Egress Inference Routing Demo Deployed${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Demo UI:${NC}        https://${DEMO_URL}"
echo -e "  ${BLUE}ExtProc GW:${NC}     http://${GW_URL}  (direct, no auth)"
echo -e "  ${BLUE}Auth GW:${NC}        http://${AUTH_GW_URL}  (Kuadrant auth)"
echo -e "  ${BLUE}BBR GW:${NC}         http://${BBR_URL}  (BBR plugin, no auth)"
echo ""
echo -e "  ${YELLOW}Namespaces:${NC}"
echo -e "    ${NAMESPACE}       — internal zone (vSR, MaaS API, demo UI)"
echo -e "    ${EXT_NAMESPACE}    — external zone (mock providers)"
echo ""
if [[ -n "$FREE_TOKEN" ]]; then
    echo -e "  ${YELLOW}Demo tokens (24h expiry):${NC}"
    echo -e "    Free user:    ${FREE_TOKEN:0:10}... (24h expiry)"
    echo -e "    Premium user: ${PREMIUM_TOKEN:0:10}... (24h expiry)"
    echo ""
    echo -e "  ${YELLOW}Quick test:${NC}"
    echo -e "    curl -H 'Authorization: Bearer <token>' http://${AUTH_GW_URL}/v1/chat/completions \\"
    echo -e "      -H 'Content-Type: application/json' \\"
    echo -e "      -d '{\"model\":\"mock-llama3\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
fi
echo ""
echo -e "  Cleanup:  $0 --cleanup"
echo ""

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
    # GPU resources
    oc delete clusterpolicy gpu-cluster-policy --ignore-not-found=true 2>/dev/null || true
    GPU_MS=$(oc get machineset -n openshift-machine-api --no-headers 2>/dev/null | grep gpu | awk '{print $1}')
    if [[ -n "$GPU_MS" ]]; then
        oc scale machineset "$GPU_MS" --replicas=0 -n openshift-machine-api 2>/dev/null || true
        oc delete machineset "$GPU_MS" -n openshift-machine-api --ignore-not-found=true 2>/dev/null || true
    fi
    # Kuadrant resources
    oc delete kuadrant kuadrant -n kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete subscription kuadrant-operator -n kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete csv -n kuadrant-system -l operators.coreos.com/kuadrant-operator.kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete operatorgroup kuadrant-operator-group -n kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete catalogsource kuadrant-operator-catalog -n kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete namespace kuadrant-system --ignore-not-found=true 2>/dev/null || true
    oc delete gatewayclass openshift-default --ignore-not-found=true 2>/dev/null || true
    oc delete clusterrolebinding demo-ui-admin --ignore-not-found=true 2>/dev/null || true
    oc delete clusterrole demo-ui-admin --ignore-not-found=true 2>/dev/null || true
    oc delete crd llminferenceservices.serving.kserve.io --ignore-not-found=true 2>/dev/null || true
    # Namespaces
    oc delete namespace vsr-demo-tier-premium --ignore-not-found=true 2>/dev/null || true
    oc delete namespace vsr-demo-tier-enterprise --ignore-not-found=true 2>/dev/null || true
    oc delete namespace "$EXT_NAMESPACE" --ignore-not-found=true 2>/dev/null || true
    oc delete namespace "$NAMESPACE" --ignore-not-found=true
    # Note: nvidia-gpu-operator, openshift-nfd left in place (reusable across deploys)
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
cp "$SCRIPT_DIR/nb-demo.html" "$BUILD_TMP/"
cp "$SCRIPT_DIR/demo-server.py" "$BUILD_TMP/"
oc start-build demo-ui --from-dir="$BUILD_TMP" --follow -n "$NAMESPACE" || true
rm -rf "$BUILD_TMP"
success "demo-ui image built"

# ─── Deploy external providers (separate namespace) ───
log "Deploying external providers (simulated external zone)..."
oc apply -f "$SCRIPT_DIR/infra/external-providers.yaml"

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

# ─── Detect cluster domain (needed for Gateway hostname + demo-config) ───
CLUSTER_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}' 2>/dev/null || echo "")
if [[ -z "$CLUSTER_DOMAIN" ]]; then
    error "Could not determine cluster domain"
fi
export CLUSTER_DOMAIN

# ─── Create demo-config (must exist before demo-ui pod starts) ───
AUTH_GW_HOST="vsr-demo.${CLUSTER_DOMAIN}"
oc create configmap demo-config \
    --from-literal=auth-gateway-host="$AUTH_GW_HOST" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

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
oc apply -f "$SCRIPT_DIR/infra/kuadrant-install.yaml"

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
oc apply -f "$SCRIPT_DIR/infra/gatewayclass.yaml"
success "GatewayClass ready"

# ─── Create Gateway ───
log "Creating Gateway..."
envsubst '${CLUSTER_DOMAIN}' < "$SCRIPT_DIR/infra/gateway.yaml" | oc apply -f -
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
oc apply -f "$SCRIPT_DIR/infra/kuadrant-cr.yaml"

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
oc apply -f "$SCRIPT_DIR/infra/maas-api.yaml"
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
oc apply -f "$SCRIPT_DIR/infra/demo-users.yaml"
success "Demo users created (free-user, premium-user)"

# ─── Apply AuthPolicy ───
log "Applying AuthPolicy..."
oc apply -f "$SCRIPT_DIR/infra/auth-policy.yaml"

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

# ─── Apply HTTPRoutes ───
log "Applying HTTPRoutes..."
oc apply -f "$SCRIPT_DIR/infra/httproute.yaml"
oc apply -f "$SCRIPT_DIR/infra/httproute-maas.yaml"
success "HTTPRoutes created"

# ─── Apply RateLimitPolicy ───
log "Applying RateLimitPolicy..."
oc apply -f "$SCRIPT_DIR/infra/ratelimit-policy.yaml"
success "RateLimitPolicy applied"

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

# =============================================================================
# GPU INFRASTRUCTURE
# =============================================================================

log ""
log "═══════════════════════════════════════════════"
log "  GPU: Setting up NVIDIA GPU node + vLLM"
log "═══════════════════════════════════════════════"

# ─── GPU MachineSet ───
CLUSTER_ID=$(oc get machineset -n openshift-machine-api --no-headers 2>/dev/null | head -1 | awk '{print $1}' | sed 's/-worker.*//')
GPU_MS="${CLUSTER_ID}-gpu-us-east-2a"
if oc get machineset "$GPU_MS" -n openshift-machine-api &>/dev/null; then
    GPU_REPLICAS=$(oc get machineset "$GPU_MS" -n openshift-machine-api -o jsonpath='{.spec.replicas}' 2>/dev/null)
    if [[ "$GPU_REPLICAS" == "0" ]]; then
        log "Scaling up existing GPU MachineSet..."
        oc scale machineset "$GPU_MS" --replicas=1 -n openshift-machine-api 2>/dev/null
    else
        success "GPU MachineSet already running"
    fi
else
    log "Creating GPU MachineSet (g5.xlarge — NVIDIA A10G, 24GB VRAM)..."
    # Get reference values from existing MachineSet
    REF_MS=$(oc get machineset -n openshift-machine-api --no-headers 2>/dev/null | head -1 | awk '{print $1}')
    AMI=$(oc get machineset "$REF_MS" -n openshift-machine-api -o jsonpath='{.spec.template.spec.providerSpec.value.ami.id}')
    IAM_PROFILE=$(oc get machineset "$REF_MS" -n openshift-machine-api -o jsonpath='{.spec.template.spec.providerSpec.value.iamInstanceProfile.id}')

    oc apply -f - <<EOF
apiVersion: machine.openshift.io/v1beta1
kind: MachineSet
metadata:
  name: ${GPU_MS}
  namespace: openshift-machine-api
  labels:
    machine.openshift.io/cluster-api-cluster: ${CLUSTER_ID}
spec:
  replicas: 1
  selector:
    matchLabels:
      machine.openshift.io/cluster-api-cluster: ${CLUSTER_ID}
      machine.openshift.io/cluster-api-machineset: ${GPU_MS}
  template:
    metadata:
      labels:
        machine.openshift.io/cluster-api-cluster: ${CLUSTER_ID}
        machine.openshift.io/cluster-api-machine-role: worker
        machine.openshift.io/cluster-api-machine-type: worker
        machine.openshift.io/cluster-api-machineset: ${GPU_MS}
    spec:
      metadata:
        labels:
          node-role.kubernetes.io/gpu: ""
      providerSpec:
        value:
          apiVersion: machine.openshift.io/v1beta1
          kind: AWSMachineProviderConfig
          ami:
            id: ${AMI}
          instanceType: g5.xlarge
          placement:
            availabilityZone: us-east-2a
            region: us-east-2
          subnet:
            filters:
            - name: tag:Name
              values:
              - ${CLUSTER_ID}-subnet-private-us-east-2a
          securityGroups:
          - filters:
            - name: tag:Name
              values:
              - ${CLUSTER_ID}-node
          - filters:
            - name: tag:Name
              values:
              - ${CLUSTER_ID}-lb
          iamInstanceProfile:
            id: ${IAM_PROFILE}
          blockDevices:
          - ebs:
              volumeSize: 120
              volumeType: gp3
          credentialsSecret:
            name: aws-cloud-credentials
          userDataSecret:
            name: worker-user-data
EOF
    success "GPU MachineSet created"
fi

# Scale CPU workers to 2 (save resources — GPU node replaces one)
CPU_MS=$(oc get machineset -n openshift-machine-api --no-headers 2>/dev/null | grep worker | grep -v gpu | awk '{print $1}')
CPU_REPLICAS=$(oc get machineset "$CPU_MS" -n openshift-machine-api -o jsonpath='{.spec.replicas}' 2>/dev/null)
if [[ "$CPU_REPLICAS" -gt 2 ]]; then
    log "Scaling CPU workers from $CPU_REPLICAS to 2..."
    oc scale machineset "$CPU_MS" --replicas=2 -n openshift-machine-api 2>/dev/null
fi

# ─── NFD + GPU Operator ───
log "Installing NFD + NVIDIA GPU operators..."
oc apply -f "$SCRIPT_DIR/infra/gpu-setup.yaml"

# Wait for NFD operator
log "Waiting for NFD operator..."
for i in $(seq 1 30); do
    CSV=$(oc get csv -n openshift-nfd --no-headers 2>/dev/null | grep "^nfd" | awk '{print $1}' | head -1)
    PHASE=$(oc get csv "$CSV" -n openshift-nfd -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "$PHASE" == "Succeeded" ]]; then success "NFD operator ready"; break; fi
    if [[ $i -eq 30 ]]; then warn "NFD operator timeout"; fi
    sleep 5
done

# Create NFD instance
log "Creating NFD instance..."
oc apply -f - <<'NFDEOF'
apiVersion: nfd.openshift.io/v1
kind: NodeFeatureDiscovery
metadata:
  name: nfd-instance
  namespace: openshift-nfd
spec:
  operand:
    servicePort: 12000
  workerConfig:
    configData: |
      core:
        sleepInterval: 60s
NFDEOF

# Wait for GPU operator
log "Waiting for GPU operator..."
for i in $(seq 1 30); do
    CSV=$(oc get csv -n nvidia-gpu-operator --no-headers 2>/dev/null | grep gpu-operator | awk '{print $1}' | head -1)
    PHASE=$(oc get csv "$CSV" -n nvidia-gpu-operator -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "$PHASE" == "Succeeded" ]]; then success "GPU operator ready"; break; fi
    if [[ $i -eq 30 ]]; then warn "GPU operator timeout"; fi
    sleep 10
done

# Create ClusterPolicy
log "Creating GPU ClusterPolicy..."
oc apply -f "$SCRIPT_DIR/infra/gpu-clusterpolicy.yaml"

# Wait for GPU node to be Ready
log "Waiting for GPU node to be Ready..."
for i in $(seq 1 40); do
    STATUS=$(oc get nodes -l node-role.kubernetes.io/gpu --no-headers 2>/dev/null | awk '{print $2}')
    if [[ "$STATUS" == "Ready" ]]; then success "GPU node Ready"; break; fi
    if [[ $i -eq 40 ]]; then warn "GPU node timeout — vLLM may deploy later"; fi
    sleep 15
done

# Wait for NFD to label the GPU node with PCI device
log "Waiting for NFD to detect GPU..."
for i in $(seq 1 30); do
    PCI=$(oc get nodes -l node-role.kubernetes.io/gpu -o jsonpath='{.items[0].metadata.labels.feature\.node\.kubernetes\.io/pci-0302_10de\.present}' 2>/dev/null)
    if [[ "$PCI" == "true" ]]; then success "NVIDIA GPU detected by NFD"; break; fi
    if [[ $i -eq 30 ]]; then warn "NFD GPU detection timeout"; fi
    sleep 10
done

# Wait for NVIDIA driver + device plugin
log "Waiting for NVIDIA driver installation (this takes several minutes)..."
for i in $(seq 1 60); do
    GPU_CAP=$(oc get nodes -l node-role.kubernetes.io/gpu -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)
    if [[ "$GPU_CAP" == "1" ]]; then success "nvidia.com/gpu: 1 — GPU fully operational"; break; fi
    if [[ $i -eq 60 ]]; then warn "GPU driver timeout — check nvidia-gpu-operator pods"; fi
    sleep 20
done

# ─── Deploy vLLM on GPU ───
log "Deploying vLLM (Qwen2.5-7B-Instruct) on GPU node..."
# vLLM needs anyuid SCC for subprocess-based engine core
oc adm policy add-scc-to-user anyuid -z default -n "$NAMESPACE" 2>/dev/null || true
oc adm policy add-scc-to-user privileged -z default -n "$NAMESPACE" 2>/dev/null || true
oc apply -n "$NAMESPACE" -f "$SCRIPT_DIR/infra/vllm-gpu.yaml"

# ─── Wait for all pods ───
log "Waiting for all pods to be ready..."
oc wait --for=condition=Ready pod -l app=mock-anthropic -n "$EXT_NAMESPACE" --timeout=120s 2>/dev/null || warn "mock-anthropic not ready yet"
oc wait --for=condition=Ready pod -l app=llm-katan,role=external-provider -n "$EXT_NAMESPACE" --timeout=120s 2>/dev/null || warn "llm-katan-external not ready yet"
oc wait --for=condition=Ready pod -l app=llm-katan,role=internal-model -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "llm-katan-internal not ready yet"
oc wait --for=condition=Ready pod -l app=vsr-router -n "$NAMESPACE" --timeout=300s 2>/dev/null || warn "vsr-router not ready yet"
oc wait --for=condition=Ready pod -l app=bbr-router -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "bbr-router not ready yet"
oc wait --for=condition=Ready pod -l app=demo-ui -n "$NAMESPACE" --timeout=120s 2>/dev/null || warn "demo-ui not ready yet"
oc wait --for=condition=Ready pod -l app=vllm-gpu -n "$NAMESPACE" --timeout=600s 2>/dev/null || warn "vllm-gpu not ready yet (model downloading)"

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
    echo -e "      -d '{\"model\":\"qwen3-0.6b\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
fi
echo ""
echo -e "  Cleanup:  $0 --cleanup"
echo ""

#!/bin/bash
#
# Deploy Jaeger Tracing to OpenShift
#
# This script deploys Jaeger all-in-one for distributed tracing visualization
# and updates the semantic-router configuration to enable tracing.
#

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

NAMESPACE="vllm-semantic-router-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}${BOLD}  Deploying Jaeger Distributed Tracing${NC}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Check if logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo -e "${RED}Error: Not logged into OpenShift${NC}"
    echo -e "${YELLOW}Please run: oc login${NC}"
    exit 1
fi

# Deploy Jaeger
echo -e "${CYAN}📊 Deploying Jaeger all-in-one...${NC}"
oc apply -f "${SCRIPT_DIR}/../observability/jaeger-deployment.yaml" -n "$NAMESPACE"

# Wait for Jaeger to be ready
echo -e "${CYAN}⏳ Waiting for Jaeger to be ready...${NC}"
oc rollout status deployment/jaeger -n "$NAMESPACE" --timeout=90s

# Get Jaeger URL
JAEGER_URL=$(oc get route jaeger -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)

if [ -z "$JAEGER_URL" ]; then
    echo -e "${RED}Warning: Could not retrieve Jaeger route${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ Jaeger deployed successfully!${NC}\n"

# Check current tracing status
CURRENT_CONFIG=$(oc get configmap semantic-router-config -n "$NAMESPACE" -o jsonpath='{.data.config\.yaml}' 2>/dev/null | grep -A 1 "tracing:" | grep "enabled:" | awk '{print $2}' || echo "false")

if [ "$CURRENT_CONFIG" = "true" ]; then
    echo -e "${GREEN}✓ Tracing is already enabled in config${NC}\n"
else
    echo -e "${YELLOW}⚠️  Tracing is currently disabled in semantic-router config${NC}"
    echo -e "${YELLOW}   To enable tracing, update the ConfigMap:${NC}\n"
    cat << EOF
${CYAN}# Edit the ConfigMap:${NC}
oc edit configmap semantic-router-config -n ${NAMESPACE}

${CYAN}# Change observability.tracing.enabled to true:${NC}
observability:
  tracing:
    enabled: true  # Change from false to true
    provider: "opentelemetry"
    exporter:
      type: "otlp"
      endpoint: "jaeger:4317"  # Jaeger OTLP gRPC endpoint
      insecure: true
    sampling:
      type: "always_on"
    resource:
      service_name: "vllm-semantic-router"
      service_version: "v0.1.0"
      deployment_environment: "openshift-demo"

${CYAN}# Then restart semantic-router to apply changes:${NC}
oc rollout restart deployment/semantic-router -n ${NAMESPACE}
EOF
    echo ""
fi

# Display URLs
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Jaeger Tracing URLs${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${CYAN}🔍 Jaeger UI:${NC}"
echo -e "   ${BOLD}https://${JAEGER_URL}${NC}\n"

echo -e "${YELLOW}💡 How to use for your demo:${NC}"
echo -e "   1. Enable tracing in semantic-router config (see above)"
echo -e "   2. Run some requests through the semantic router"
echo -e "   3. Open Jaeger UI and select 'vllm-semantic-router' service"
echo -e "   4. Click 'Find Traces' to see request flows"
echo -e "   5. Click on a trace to see the detailed span timeline\n"

echo -e "${CYAN}📋 What you'll see in traces:${NC}"
echo -e "   • Request ingress through Envoy"
echo -e "   • ExtProc classification pipeline"
echo -e "   • Security checks (jailbreak, PII)"
echo -e "   • Category classification"
echo -e "   • Model routing decisions"
echo -e "   • Cache hits/misses"
echo -e "   • End-to-end latency breakdown\n"

echo -e "${GREEN}${BOLD}Ready for your demo! 🎉${NC}\n"

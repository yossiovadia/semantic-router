#!/usr/bin/env bash

# Complete OpenShift Demo Setup
#
# Deploys everything needed for a comprehensive semantic router demo:
# - Jaeger distributed tracing
# - Flow visualization (interactive HTML)
# - Unified dashboard (all-in-one UI with sidebar)
# - Enables tracing in semantic-router config

set -euo pipefail

NAMESPACE="${NAMESPACE:-vllm-semantic-router-system}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
BLUE='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}  Semantic Router Demo Setup${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if we're logged in to OpenShift
if ! oc whoami &>/dev/null; then
  echo -e "${YELLOW}⚠️  Not logged in to OpenShift. Please run 'oc login' first.${NC}"
  exit 1
fi

echo -e "${GREEN}Current cluster: $(oc cluster-info | head -1)${NC}"
echo -e "${GREEN}Current namespace: ${NAMESPACE}${NC}"
echo ""

# ==============================================================================
# 1. Deploy Jaeger
# ==============================================================================
echo -e "${BLUE}${BOLD}[1/4] Deploying Jaeger distributed tracing...${NC}"

oc apply -f "${SCRIPT_DIR}/../observability/jaeger-deployment.yaml" 2>/dev/null || true

echo -e "${BLUE}⏳ Waiting for Jaeger to be ready...${NC}"
oc wait --for=condition=available --timeout=120s deployment/jaeger -n "${NAMESPACE}" 2>/dev/null || {
  echo -e "${YELLOW}⚠️  Jaeger deployment taking longer than expected, continuing...${NC}"
}

echo -e "${GREEN}✅ Jaeger deployed${NC}"
echo ""

# ==============================================================================
# 2. Deploy Flow Visualization
# ==============================================================================
echo -e "${BLUE}${BOLD}[2/4] Deploying flow visualization...${NC}"

# Create ConfigMap from HTML file
oc create configmap flow-visualization-html \
  --from-file=index.html="${SCRIPT_DIR}/flow-visualization.html" \
  --dry-run=client -o yaml | oc apply -f - -n "${NAMESPACE}"

# Apply deployment manifests
oc apply -f "${SCRIPT_DIR}/flow-viz-deployment.yaml"

# Restart deployment to pick up new ConfigMap
oc rollout restart deployment/flow-visualization -n "${NAMESPACE}" 2>/dev/null || true

echo -e "${BLUE}⏳ Waiting for flow-visualization to be ready...${NC}"
oc wait --for=condition=available --timeout=90s deployment/flow-visualization -n "${NAMESPACE}" 2>/dev/null || {
  echo -e "${YELLOW}⚠️  Flow visualization deployment taking longer, continuing...${NC}"
}

echo -e "${GREEN}✅ Flow visualization deployed${NC}"
echo ""

# ==============================================================================
# 3. Deploy Unified Dashboard
# ==============================================================================
echo -e "${BLUE}${BOLD}[3/4] Deploying unified dashboard...${NC}"

oc apply -f "${SCRIPT_DIR}/../dashboard/dashboard-deployment.yaml"

echo -e "${BLUE}⏳ Waiting for dashboard to be ready...${NC}"
oc wait --for=condition=available --timeout=180s deployment/dashboard -n "${NAMESPACE}" 2>/dev/null || {
  echo -e "${YELLOW}⚠️  Dashboard deployment taking longer, continuing...${NC}"
}

echo -e "${GREEN}✅ Dashboard deployed${NC}"
echo ""

# ==============================================================================
# 4. Enable Tracing
# ==============================================================================
echo -e "${BLUE}${BOLD}[4/4] Enabling distributed tracing...${NC}"

# Check current status
CURRENT_STATUS=$(oc get configmap semantic-router-config -n "${NAMESPACE}" -o yaml | grep -A 1 "tracing:" | grep "enabled:" | awk '{print $2}')

if [ "${CURRENT_STATUS}" = "true" ]; then
  echo -e "${GREEN}✓ Tracing already enabled${NC}"
else
  echo -e "${BLUE}Enabling tracing in semantic-router config...${NC}"

  # Enable tracing
  oc get configmap semantic-router-config -n "${NAMESPACE}" -o yaml | \
    sed 's/enabled: false  # Enable distributed tracing/enabled: true  # Enable distributed tracing/' | \
    oc replace -f -

  echo -e "${BLUE}Restarting semantic-router to apply changes...${NC}"
  oc rollout restart deployment/semantic-router -n "${NAMESPACE}"

  echo -e "${BLUE}⏳ Waiting for semantic-router to restart...${NC}"
  sleep 10  # Give it a moment to start the rollout
  oc rollout status deployment/semantic-router -n "${NAMESPACE}" --timeout=180s 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Semantic-router restart taking longer, check status with: oc get pods -n ${NAMESPACE}${NC}"
  }

  echo -e "${GREEN}✅ Tracing enabled${NC}"
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Demo Setup Complete! 🎉${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Get all the URLs
DASHBOARD_URL=$(oc get route dashboard -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
JAEGER_URL=$(oc get route jaeger -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
FLOW_VIZ_URL=$(oc get route flow-visualization -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
GRAFANA_URL=$(oc get route grafana -n "${NAMESPACE}" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

echo -e "${BLUE}🌐 Access URLs:${NC}"
echo ""

if [ -n "${DASHBOARD_URL}" ]; then
  echo -e "${GREEN}${BOLD}★ UNIFIED DASHBOARD (START HERE):${NC}"
  echo -e "   https://${DASHBOARD_URL}"
  echo -e "   ${BLUE}├─ 🎮 Playground (OpenWebUI)${NC}"
  echo -e "   ${BLUE}├─ 🤖 Models Configuration${NC}"
  echo -e "   ${BLUE}├─ 🛡️  Prompt Guard${NC}"
  echo -e "   ${BLUE}├─ ⚡ Similarity Cache${NC}"
  echo -e "   ${BLUE}├─ 🧠 Intelligent Routing${NC}"
  echo -e "   ${BLUE}├─ 🗺️  Topology Visualization${NC}"
  echo -e "   ${BLUE}├─ 🔧 Tools Selection${NC}"
  echo -e "   ${BLUE}├─ 👁️  Observability${NC}"
  echo -e "   ${BLUE}├─ 🔌 Classification API${NC}"
  echo -e "   ${BLUE}└─ 📊 Monitoring (Grafana embedded)${NC}"
  echo ""
fi

echo -e "${BLUE}Individual Service URLs:${NC}"
[ -n "${JAEGER_URL}" ] && echo -e "   🔍 Jaeger Tracing:        https://${JAEGER_URL}"
[ -n "${FLOW_VIZ_URL}" ] && echo -e "   🌊 Flow Visualization:    https://${FLOW_VIZ_URL}"
[ -n "${GRAFANA_URL}" ] && echo -e "   📊 Grafana Direct:        https://${GRAFANA_URL}"

echo ""
echo -e "${YELLOW}💡 Quick Demo Commands:${NC}"
echo ""
echo -e "   # Run test classifications"
echo -e "   ./deploy/openshift/demo/curl-examples.sh all"
echo ""
echo -e "   # Or use interactive Python demo"
echo -e "   python3 deploy/openshift/demo/demo-semantic-router.py"
echo ""
echo -e "   # View live logs"
echo -e "   ./deploy/openshift/demo/live-semantic-router-logs.sh"
echo ""

echo -e "${GREEN}${BOLD}Ready to present! 🚀${NC}"
echo ""
echo -e "${BLUE}For detailed demo flow, see: deploy/openshift/demo/DEMO-README.md${NC}"

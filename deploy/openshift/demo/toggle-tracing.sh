#!/bin/bash
#
# Toggle Tracing On/Off for Semantic Router
#
# Usage:
#   ./toggle-tracing.sh enable   # Enable tracing
#   ./toggle-tracing.sh disable  # Disable tracing
#   ./toggle-tracing.sh status   # Check current status
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
ACTION="${1:-status}"

# Check if logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo -e "${RED}Error: Not logged into OpenShift${NC}"
    echo -e "${YELLOW}Please run: oc login${NC}"
    exit 1
fi

# Function to check current status
check_status() {
    local enabled=$(oc get configmap semantic-router-config -n "$NAMESPACE" -o jsonpath='{.data.config\.yaml}' 2>/dev/null | grep -A 1 "tracing:" | grep "enabled:" | awk '{print $2}')

    if [ "$enabled" = "true" ]; then
        echo -e "${GREEN}✓ Tracing is ENABLED${NC}"

        # Check if Jaeger is running
        if oc get deployment jaeger -n "$NAMESPACE" &>/dev/null; then
            JAEGER_URL=$(oc get route jaeger -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
            echo -e "${CYAN}  Jaeger UI: https://${JAEGER_URL}${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Jaeger is not deployed. Run: ./deploy-jaeger.sh${NC}"
        fi
    else
        echo -e "${YELLOW}✗ Tracing is DISABLED${NC}"
    fi
}

# Function to enable tracing
enable_tracing() {
    echo -e "${CYAN}Enabling tracing...${NC}"

    # Check if Jaeger is deployed
    if ! oc get deployment jaeger -n "$NAMESPACE" &>/dev/null; then
        echo -e "${YELLOW}⚠️  Jaeger is not deployed. Deploying now...${NC}"
        ./deploy/openshift/demo/deploy-jaeger.sh
    fi

    # Update ConfigMap
    oc get configmap semantic-router-config -n "$NAMESPACE" -o yaml | \
        sed 's/enabled: false  # Enable distributed tracing/enabled: true  # Enable distributed tracing/' | \
        sed 's/type: "stdout"/type: "otlp"/' | \
        sed 's/endpoint: "localhost:4317"/endpoint: "jaeger:4317"/' | \
        sed 's/deployment_environment: "development"/deployment_environment: "openshift-demo"/' | \
        oc apply -f -

    # Restart semantic-router
    echo -e "${CYAN}Restarting semantic-router...${NC}"
    oc rollout restart deployment/semantic-router -n "$NAMESPACE"
    oc rollout status deployment/semantic-router -n "$NAMESPACE" --timeout=60s

    echo -e "${GREEN}✓ Tracing enabled successfully!${NC}\n"
    check_status
}

# Function to disable tracing
disable_tracing() {
    echo -e "${CYAN}Disabling tracing...${NC}"

    # Update ConfigMap
    oc get configmap semantic-router-config -n "$NAMESPACE" -o yaml | \
        sed 's/enabled: true  # Enable distributed tracing/enabled: false  # Enable distributed tracing/' | \
        oc apply -f -

    # Restart semantic-router
    echo -e "${CYAN}Restarting semantic-router...${NC}"
    oc rollout restart deployment/semantic-router -n "$NAMESPACE"
    oc rollout status deployment/semantic-router -n "$NAMESPACE" --timeout=60s

    echo -e "${GREEN}✓ Tracing disabled${NC}\n"
    check_status
}

# Main logic
case "$ACTION" in
    enable)
        enable_tracing
        ;;
    disable)
        disable_tracing
        ;;
    status)
        check_status
        ;;
    *)
        echo -e "${BOLD}Usage:${NC}"
        echo -e "  $0 enable   # Enable distributed tracing"
        echo -e "  $0 disable  # Disable distributed tracing"
        echo -e "  $0 status   # Check current status"
        echo ""
        check_status
        exit 1
        ;;
esac

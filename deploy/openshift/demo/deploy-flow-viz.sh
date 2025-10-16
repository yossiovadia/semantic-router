#!/bin/bash
#
# Deploy Flow Visualization to OpenShift
#
# This script deploys the interactive flow visualization HTML page
# to OpenShift as a lightweight nginx service with a public route.
#

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="vllm-semantic-router-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="${SCRIPT_DIR}/flow-visualization.html"

echo -e "${CYAN}Deploying Flow Visualization to OpenShift...${NC}\n"

# Check if logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo -e "${RED}Error: Not logged into OpenShift${NC}"
    echo -e "${YELLOW}Please run: oc login${NC}"
    exit 1
fi

# Check if HTML file exists
if [ ! -f "$HTML_FILE" ]; then
    echo -e "${RED}Error: flow-visualization.html not found${NC}"
    exit 1
fi

# Create or update ConfigMap with the HTML content
echo -e "${CYAN}📄 Creating ConfigMap with HTML content...${NC}"
oc create configmap flow-visualization-html \
    --from-file=index.html="$HTML_FILE" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | oc apply -f -

# Deploy the nginx service
echo -e "${CYAN}🚀 Deploying nginx service...${NC}"
oc apply -f "${SCRIPT_DIR}/flow-viz-deployment.yaml" -n "$NAMESPACE"

# Wait for deployment to be ready
echo -e "${CYAN}⏳ Waiting for deployment to be ready...${NC}"
oc rollout status deployment/flow-visualization -n "$NAMESPACE" --timeout=60s

# Get the route URL
echo -e "\n${GREEN}✅ Deployment successful!${NC}\n"

ROUTE_URL=$(oc get route flow-visualization -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)

if [ -n "$ROUTE_URL" ]; then
    echo -e "${GREEN}🌐 Flow Visualization URL:${NC}"
    echo -e "   ${CYAN}http://${ROUTE_URL}${NC}\n"

    echo -e "${YELLOW}💡 For your demo:${NC}"
    echo -e "   1. Open the URL in your browser"
    echo -e "   2. Click 'Start Animation' to show the flow"
    echo -e "   3. Click any step to see detailed information\n"
else
    echo -e "${RED}Warning: Could not retrieve route URL${NC}"
    echo -e "${YELLOW}Check manually with: oc get route flow-visualization -n ${NAMESPACE}${NC}\n"
fi

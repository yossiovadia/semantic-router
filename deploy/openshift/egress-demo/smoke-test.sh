#!/usr/bin/env bash
# =============================================================================
# vSR Egress Demo — Smoke Test
#
# Validates all demo scenarios against expected behavior.
# Works with both local (make run-egress-demo) and OpenShift deployments.
#
# Usage:
#   ./smoke-test.sh                 # Auto-detect: OpenShift route or localhost
#   ./smoke-test.sh http://gw:8801  # Explicit gateway URL
#   ./smoke-test.sh --phase 1       # Phase 1 tests only (egress MVP)
#   ./smoke-test.sh --phase 2       # + Phase 2 (BBR plugin, expected failures)
#   ./smoke-test.sh --phase 3       # + Phase 3 (MaaS/Kuadrant auth via SA tokens)
#   ./smoke-test.sh --phase 3 --gateway-url https://vsr-demo-gateway-istio-...  # Explicit
#
# Phases:
#   1 = Egress routing, API translation, tier access (should pass now)
#   2 = BBR plugin integration (expected to fail until Phase 2 is done)
#   3 = Full stack with MaaS + Kuadrant (real SA token auth via oc create token)
# =============================================================================

set -uo pipefail

GATEWAY_URL=""
GATEWAY_AUTH_URL=""
PHASE="1"
PASS=0
FAIL=0
SKIP=0
EXPECTED_FAIL=0

GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --gateway-url) GATEWAY_AUTH_URL="$2"; shift 2 ;;
        http*) GATEWAY_URL="$1"; shift ;;
        *) shift ;;
    esac
done

# Auto-detect gateway URL if not provided
if [[ -z "$GATEWAY_URL" ]]; then
    # Try OpenShift route
    if command -v oc &>/dev/null && oc whoami &>/dev/null 2>&1; then
        GW_HOST=$(oc get route gateway-route -n vsr-egress-demo -o jsonpath='{.spec.host}' 2>/dev/null || true)
        if [[ -n "$GW_HOST" ]]; then
            GATEWAY_URL="http://${GW_HOST}"
            echo -e "${GREEN}Auto-detected${RESET} OpenShift gateway: $GATEWAY_URL"
        fi
    fi
    # Fallback to localhost
    if [[ -z "$GATEWAY_URL" ]]; then
        GATEWAY_URL="http://localhost:8801"
        echo -e "${YELLOW}No OpenShift route found${RESET}, using local: $GATEWAY_URL"
    fi
fi

assert_response() {
    local test_name="$1"
    local response="$2"
    local field="$3"
    local expected="$4"
    local expect_fail="${5:-false}"

    local actual
    actual=$(echo "$response" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    # Navigate dotted path: error.code, choices.0.message.role, etc.
    parts = '$field'.split('.')
    v = d
    for p in parts:
        if p.isdigit():
            v = v[int(p)]
        else:
            v = v[p]
    print(v)
except:
    print('MISSING')
" 2>/dev/null || echo "PARSE_ERROR")

    if [[ "$actual" == "$expected" ]]; then
        if [[ "$expect_fail" == "true" ]]; then
            echo -e "  ${YELLOW}UNEXPECTED PASS${RESET} $test_name: $field=$actual (expected to fail)"
            ((PASS++))
        else
            echo -e "  ${GREEN}PASS${RESET} $test_name: $field=$actual"
            ((PASS++))
        fi
    else
        if [[ "$expect_fail" == "true" ]]; then
            echo -e "  ${BLUE}EXPECTED FAIL${RESET} $test_name: $field=$actual (expected: $expected)"
            ((EXPECTED_FAIL++))
        else
            echo -e "  ${RED}FAIL${RESET} $test_name: $field=$actual (expected: $expected)"
            ((FAIL++))
        fi
    fi
}

assert_header() {
    local test_name="$1"
    local headers_file="$2"
    local header_name="$3"
    local expected="$4"
    local expect_fail="${5:-false}"

    local actual
    actual=$(grep -i "^${header_name}:" "$headers_file" 2>/dev/null | sed "s/^${header_name}: *//i" | tr -d '\r' || echo "MISSING")

    if [[ -z "$actual" ]]; then
        actual="MISSING"
    fi

    if [[ "$actual" == "$expected" ]]; then
        if [[ "$expect_fail" == "true" ]]; then
            echo -e "  ${YELLOW}UNEXPECTED PASS${RESET} $test_name: header $header_name=$actual"
            ((PASS++))
        else
            echo -e "  ${GREEN}PASS${RESET} $test_name: header $header_name=$actual"
            ((PASS++))
        fi
    else
        if [[ "$expect_fail" == "true" ]]; then
            echo -e "  ${BLUE}EXPECTED FAIL${RESET} $test_name: header $header_name=$actual (expected: $expected)"
            ((EXPECTED_FAIL++))
        else
            echo -e "  ${RED}FAIL${RESET} $test_name: header $header_name=$actual (expected: $expected)"
            ((FAIL++))
        fi
    fi
}

assert_status() {
    local test_name="$1"
    local headers_file="$2"
    local expected="$3"

    local actual
    actual=$(head -1 "$headers_file" | grep -oE "[0-9]{3}" | head -1 || echo "UNKNOWN")

    if [[ "$actual" == "$expected" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $test_name: HTTP $actual"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $test_name: HTTP $actual (expected: $expected)"
        ((FAIL++))
    fi
}

HEADERS=$(mktemp)
BODY=$(mktemp)
trap "rm -f $HEADERS $BODY" EXIT

echo ""
echo "================================================================"
echo "  vSR Egress Demo — Smoke Test"
echo "  Gateway:      $GATEWAY_URL"
if [[ -n "$GATEWAY_AUTH_URL" ]]; then
echo "  Gateway Auth: $GATEWAY_AUTH_URL"
fi
echo "  Phase:        $PHASE"
echo "================================================================"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Egress Routing MVP
# ═══════════════════════════════════════════════════════════════
echo "--- Phase 1: Egress Routing MVP ---"
echo ""

# Test 1.1: External provider routing
echo "Test 1.1: External Provider Routing"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
assert_status "External routing" "$HEADERS" "200"
assert_response "External routing" "$(cat $BODY)" "model" "qwen2.5:1.5b"
assert_header "External routing" "$HEADERS" "x-vsr-selected-model" "qwen2.5:1.5b"
echo ""

# Test 1.2: API translation (Anthropic)
echo "Test 1.2: API Translation (Anthropic)"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"claude-sonnet","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
assert_status "API translation" "$HEADERS" "200"
assert_header "API translation" "$HEADERS" "x-vsr-selected-model" "claude-sonnet"
# Response should be in OpenAI format (translated from Anthropic)
assert_response "API translation" "$(cat $BODY)" "choices.0.message.role" "assistant"
echo ""

# Test 1.3: Internal model routing
echo "Test 1.3: Internal Model Routing"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mock-llama3","messages":[{"role":"user","content":"Hello"}]}'
assert_status "Internal routing" "$HEADERS" "200"
assert_response "Internal routing" "$(cat $BODY)" "model" "mock-llama3"
assert_header "Internal routing" "$HEADERS" "x-vsr-selected-model" "mock-llama3"
echo ""

# Test 1.4: Tier — free user blocked from external
echo "Test 1.4: Tier — Free User Blocked from External"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-MaaS-Tier: free" \
    -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
assert_status "Free blocked" "$HEADERS" "200"
assert_response "Free blocked" "$(cat $BODY)" "error.code" "403"
echo ""

# Test 1.5: Tier — free user allowed internal
echo "Test 1.5: Tier — Free User Allowed Internal"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-MaaS-Tier: free" \
    -d '{"model":"mock-llama3","messages":[{"role":"user","content":"Hello"}]}'
assert_status "Free allowed" "$HEADERS" "200"
assert_response "Free allowed" "$(cat $BODY)" "model" "mock-llama3"
echo ""

# Test 1.6: Tier — premium user allowed external
echo "Test 1.6: Tier — Premium User Allowed External"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-MaaS-Tier: premium" \
    -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
assert_status "Premium allowed" "$HEADERS" "200"
assert_response "Premium allowed" "$(cat $BODY)" "model" "qwen2.5:1.5b"
echo ""

# Test 1.7: Tier — no tier header = unrestricted
echo "Test 1.7: Tier — No Tier Header = Unrestricted"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
assert_status "No tier" "$HEADERS" "200"
assert_response "No tier" "$(cat $BODY)" "model" "qwen2.5:1.5b"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 2: BBR Plugin (expected to fail until Phase 2 is implemented)
# ═══════════════════════════════════════════════════════════════
if [[ "$PHASE" -ge 2 ]]; then
    echo "--- Phase 2: BBR Plugin (expected failures) ---"
    echo "(These tests validate the BBR plugin integration.)"
    echo "(They are expected to fail until Phase 2 is implemented.)"
    echo ""

    # Test 2.1: BBR plugin processes request
    echo "Test 2.1: BBR Plugin — Model Extraction"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} BBR plugin not yet deployed"
    ((EXPECTED_FAIL++))
    echo ""

    # Test 2.2: BBR plugin API translation
    echo "Test 2.2: BBR Plugin — API Translation"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} BBR plugin not yet deployed"
    ((EXPECTED_FAIL++))
    echo ""

    # Test 2.3: BBR plugin classification headers
    echo "Test 2.3: BBR Plugin — Classification Headers"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} BBR plugin not yet deployed"
    ((EXPECTED_FAIL++))
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Full Stack on OpenShift (real SA token auth)
# ═══════════════════════════════════════════════════════════════
if [[ "$PHASE" -ge 3 ]]; then
    echo "--- Phase 3: Full Stack (SA Token Auth) ---"
    echo "(These tests validate real authentication via oc create token.)"
    echo ""

    # Auto-detect gateway URL for Phase 3 if not provided
    if [[ -z "$GATEWAY_AUTH_URL" ]]; then
        if command -v oc &>/dev/null && oc whoami &>/dev/null 2>&1; then
            # Construct from Gateway listener hostname (vsr-demo.<cluster-domain>)
            CLUSTER_DOMAIN=$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true)
            if [[ -n "$CLUSTER_DOMAIN" ]]; then
                GATEWAY_AUTH_URL="http://vsr-demo.${CLUSTER_DOMAIN}"
                echo -e "${GREEN}Auto-detected${RESET} Gateway Auth URL: $GATEWAY_AUTH_URL"
            fi
        fi

        if [[ -z "$GATEWAY_AUTH_URL" ]]; then
            echo -e "${RED}ERROR${RESET}: Phase 3 requires --gateway-url <url> (Gateway API endpoint with auth)"
            echo "  Example: ./smoke-test.sh --phase 3 --gateway-url http://vsr-demo.apps.cluster.example.com"
            echo ""
            echo "  Auto-detection failed. Provide the URL explicitly or ensure 'oc' is logged in."
            ((FAIL++))
        fi
    fi

    if [[ -n "$GATEWAY_AUTH_URL" ]]; then
        echo "  Gateway Auth URL: $GATEWAY_AUTH_URL"
        echo ""

        # Test 3.1: Unauthenticated request → 401
        echo "Test 3.1: Unauthenticated Request → 401"
        curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"mock-llama3","messages":[{"role":"user","content":"Hello"}]}'
        assert_status "Unauthenticated" "$HEADERS" "401"
        echo ""

        # Test 3.2: Free user + internal model → 200
        echo "Test 3.2: Free User + Internal Model → 200"
        FREE_TOKEN=$(oc create token free-user -n vsr-egress-demo --audience vsr-demo-gateway-sa 2>/dev/null || echo "")
        if [[ -z "$FREE_TOKEN" ]]; then
            echo -e "  ${RED}FAIL${RESET} Could not create token for free-user in vsr-egress-demo"
            ((FAIL++))
        else
            curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${FREE_TOKEN}" \
                -d '{"model":"mock-llama3","messages":[{"role":"user","content":"Hello"}]}'
            assert_status "Free+internal" "$HEADERS" "200"
            assert_response "Free+internal" "$(cat $BODY)" "model" "mock-llama3"
        fi
        echo ""

        # Test 3.3: Free user + external model → blocked (403 in body)
        echo "Test 3.3: Free User + External Model → Blocked"
        if [[ -z "$FREE_TOKEN" ]]; then
            echo -e "  ${RED}FAIL${RESET} No free-user token (skipped)"
            ((FAIL++))
        else
            curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${FREE_TOKEN}" \
                -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
            assert_status "Free+external HTTP" "$HEADERS" "200"
            assert_response "Free+external body" "$(cat $BODY)" "error.code" "403"
        fi
        echo ""

        # Test 3.4: Premium user + external model → 200
        echo "Test 3.4: Premium User + External Model → 200"
        PREMIUM_TOKEN=$(oc create token premium-user -n vsr-demo-tier-premium --audience vsr-demo-gateway-sa 2>/dev/null || echo "")
        if [[ -z "$PREMIUM_TOKEN" ]]; then
            echo -e "  ${RED}FAIL${RESET} Could not create token for premium-user in vsr-demo-tier-premium"
            ((FAIL++))
        else
            curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${PREMIUM_TOKEN}" \
                -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Hello"}]}'
            assert_status "Premium+external" "$HEADERS" "200"
            assert_response "Premium+external" "$(cat $BODY)" "model" "qwen2.5:1.5b"
        fi
        echo ""

        # Test 3.5: Premium user + Anthropic model → 200 (translated)
        echo "Test 3.5: Premium User + Anthropic → 200 (Translated)"
        if [[ -z "$PREMIUM_TOKEN" ]]; then
            echo -e "  ${RED}FAIL${RESET} No premium-user token (skipped)"
            ((FAIL++))
        else
            curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${PREMIUM_TOKEN}" \
                -d '{"model":"claude-sonnet","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
            assert_status "Premium+anthropic" "$HEADERS" "200"
            assert_response "Premium+anthropic" "$(cat $BODY)" "choices.0.message.role" "assistant"
        fi
        echo ""

        # Test 3.6: Invalid token → 401
        echo "Test 3.6: Invalid Token → 401"
        curl -skS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer invalid-token" \
            -d '{"model":"mock-llama3","messages":[{"role":"user","content":"Hello"}]}'
        assert_status "Invalid token" "$HEADERS" "401"
        echo ""

        # Test 3.7: MaaS API health check (via oc exec)
        echo "Test 3.7: MaaS API Health Check"
        MAAS_POD=$(oc get pods -n vsr-egress-demo -l app=maas-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [[ -z "$MAAS_POD" ]]; then
            echo -e "  ${RED}FAIL${RESET} No maas-api pod found in vsr-egress-demo"
            ((FAIL++))
        else
            HEALTH_STATUS=$(oc exec "$MAAS_POD" -n vsr-egress-demo -- curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/health 2>/dev/null || echo "000")
            if [[ "$HEALTH_STATUS" == "200" ]]; then
                echo -e "  ${GREEN}PASS${RESET} MaaS API health: HTTP $HEALTH_STATUS"
                ((PASS++))
            else
                echo -e "  ${RED}FAIL${RESET} MaaS API health: HTTP $HEALTH_STATUS (expected: 200)"
                ((FAIL++))
            fi
        fi
        echo ""
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo "================================================================"
echo -e "  Results: ${GREEN}$PASS passed${RESET}, ${RED}$FAIL failed${RESET}, ${BLUE}$EXPECTED_FAIL expected failures${RESET}"
echo "================================================================"
echo ""

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0

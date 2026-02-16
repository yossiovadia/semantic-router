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
#   ./smoke-test.sh --phase 3       # + Phase 3 (MaaS/Kuadrant, expected failures)
#
# Phases:
#   1 = Egress routing, API translation, tier access (should pass now)
#   2 = BBR plugin integration (expected to fail until Phase 2 is done)
#   3 = Full stack with MaaS + Kuadrant (expected to fail until Phase 3 is done)
# =============================================================================

set -uo pipefail

GATEWAY_URL=""
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
echo "  Gateway: $GATEWAY_URL"
echo "  Phase:   $PHASE"
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
# PHASE 3: Full Stack on OpenShift (expected to fail until Phase 3)
# ═══════════════════════════════════════════════════════════════
if [[ "$PHASE" -ge 3 ]]; then
    echo "--- Phase 3: Full Stack (expected failures) ---"
    echo "(These tests validate MaaS + Kuadrant integration.)"
    echo "(They are expected to fail until Phase 3 is implemented.)"
    echo ""

    # Test 3.1: MaaS token issuance
    echo "Test 3.1: MaaS — Token Issuance"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} MaaS API not deployed"
    ((EXPECTED_FAIL++))
    echo ""

    # Test 3.2: Kuadrant AuthPolicy enforcement
    echo "Test 3.2: Kuadrant — AuthPolicy Enforcement"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} Kuadrant not deployed"
    ((EXPECTED_FAIL++))
    echo ""

    # Test 3.3: Kuadrant RateLimitPolicy enforcement
    echo "Test 3.3: Kuadrant — RateLimitPolicy Enforcement"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} Kuadrant not deployed"
    ((EXPECTED_FAIL++))
    echo ""

    # Test 3.4: Real auth with tier from MaaS
    echo "Test 3.4: MaaS — Tier Resolution via AuthPolicy"
    echo -e "  ${BLUE}EXPECTED FAIL${RESET} MaaS + Kuadrant not deployed"
    ((EXPECTED_FAIL++))
    echo ""
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

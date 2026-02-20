#!/usr/bin/env bash
# =============================================================================
# NB Demo — Smoke Test (TDD)
#
# Tests for the financial services demo requirements:
#   1. Department-based access (Intern/Finance/Principal Engineer)
#   2. PII detection → force internal routing
#   3. Complexity-based routing
#   4. Classification headers in responses
#
# Usage:
#   ./smoke-test-nb.sh http://<gateway-url> --gateway-url http://<auth-gateway-url>
#
# These tests define the EXPECTED behavior. Some will fail until
# classification is enabled and decision rules are configured.
# =============================================================================

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

GATEWAY_URL="${1:-}"
GATEWAY_AUTH_URL=""
PASS=0
FAIL=0
SKIP=0

while [[ $# -gt 1 ]]; do
    case $2 in
        --gateway-url) GATEWAY_AUTH_URL="$3"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$GATEWAY_URL" ]]; then
    echo "Usage: $0 <gateway-url> --gateway-url <auth-gateway-url>"
    exit 1
fi

# Auto-detect auth gateway
if [[ -z "$GATEWAY_AUTH_URL" ]]; then
    if command -v oc &>/dev/null && oc whoami &>/dev/null 2>&1; then
        CLUSTER_DOMAIN=$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true)
        if [[ -n "$CLUSTER_DOMAIN" ]]; then
            GATEWAY_AUTH_URL="http://vsr-demo.${CLUSTER_DOMAIN}"
        fi
    fi
fi

HEADERS=$(mktemp)
BODY=$(mktemp)
trap "rm -f $HEADERS $BODY" EXIT

echo ""
echo "================================================================"
echo "  NB Demo — Smoke Test (TDD)"
echo "  Gateway:      $GATEWAY_URL"
echo "  Gateway Auth: ${GATEWAY_AUTH_URL:-NOT SET}"
echo "================================================================"
echo ""

# ── Helpers ──────────────────────────────────────────

assert_status() {
    local label=$1 headers=$2 expected=$3
    local actual
    actual=$(head -1 "$headers" | grep -o '[0-9]\{3\}' | head -1)
    if [[ "$actual" == "$expected" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: HTTP $actual"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: HTTP ${actual:-UNKNOWN} (expected: $expected)"
        ((FAIL++))
    fi
}

assert_header() {
    local label=$1 headers=$2 name=$3 expected=$4
    local actual
    actual=$(grep -i "^${name}:" "$headers" | head -1 | sed 's/^[^:]*: *//' | tr -d '\r')
    if [[ "$actual" == "$expected" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: $name=$actual"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: $name=${actual:-MISSING} (expected: $expected)"
        ((FAIL++))
    fi
}

assert_header_contains() {
    local label=$1 headers=$2 name=$3 substring=$4
    local actual
    actual=$(grep -i "^${name}:" "$headers" | head -1 | sed 's/^[^:]*: *//' | tr -d '\r')
    if [[ "$actual" == *"$substring"* ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: $name contains '$substring' (value: $actual)"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: $name=${actual:-MISSING} (expected to contain: $substring)"
        ((FAIL++))
    fi
}

assert_header_exists() {
    local label=$1 headers=$2 name=$3
    local actual
    actual=$(grep -i "^${name}:" "$headers" | head -1 | sed 's/^[^:]*: *//' | tr -d '\r')
    if [[ -n "$actual" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: $name present (value: $actual)"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: $name NOT present"
        ((FAIL++))
    fi
}

assert_response() {
    local label=$1 body=$2 path=$3 expected=$4
    local actual
    actual=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$(echo $path | sed 's/\./][\"/' | sed 's/^/[\"/' | sed 's/$/\"]/'))" 2>/dev/null || echo "MISSING")
    if [[ "$actual" == "$expected" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: $path=$actual"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: $path=$actual (expected: $expected)"
        ((FAIL++))
    fi
}

assert_model_is_internal() {
    local label=$1 body=$2
    local model
    model=$(python3 -c "import sys,json; d=json.load(open('$body')); print(d.get('model',''))" 2>/dev/null || echo "")
    if [[ "$model" == "qwen3-0.6b" || "$model" == "qwen2.5-7b" ]]; then
        echo -e "  ${GREEN}PASS${RESET} $label: routed to internal model ($model)"
        ((PASS++))
    else
        echo -e "  ${RED}FAIL${RESET} $label: routed to $model (expected internal: qwen3-0.6b or qwen2.5-7b)"
        ((FAIL++))
    fi
}

# =============================================================================
# TEST GROUP 1: Classification Headers
# vSR should return classification headers when classifiers are enabled
# =============================================================================

echo "--- Group 1: Classification Headers ---"

echo "Test 1.1: Domain classification header present"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"What is the derivative of x squared?"}],"max_tokens":10}'
assert_header_exists "Domain classification" "$HEADERS" "x-vsr-matched-domains"

echo ""
echo "Test 1.2: Math query classified as math/STEM domain"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Prove that the square root of 2 is irrational using proof by contradiction"}],"max_tokens":10}'
assert_header_contains "Math classification" "$HEADERS" "x-vsr-matched-domains" "math"

echo ""

# =============================================================================
# TEST GROUP 2: PII Detection
# Queries with PII should be detected and routed to internal models only
# =============================================================================

echo "--- Group 2: PII Detection ---"

echo "Test 2.1: Query with SSN detected as PII"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"My SSN is 123-45-6789, can you help me file taxes?"}],"max_tokens":10}'
# PII detection should be indicated in headers or routing decision
assert_status "PII query" "$HEADERS" "200"

echo ""
echo "Test 2.2: PII query routed to internal model (data stays on-prem)"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"My credit card number is 4532-1234-5678-9012 and I need to check my balance"}],"max_tokens":10}'
assert_model_is_internal "PII → internal" "$BODY"

echo ""
echo "Test 2.3: Non-PII query can route to any model"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"What are the current market trends in technology?"}],"max_tokens":10}'
assert_status "Non-PII query" "$HEADERS" "200"

echo ""

# =============================================================================
# TEST GROUP 3: Complexity-Based Routing
# Simple queries → internal, Complex queries → external (when tier allows)
# =============================================================================

echo "--- Group 3: Complexity-Based Routing ---"

echo "Test 3.1: Simple query → routed to internal model (low complexity)"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}'
assert_model_is_internal "Simple math → internal" "$BODY"

echo ""
echo "Test 3.1b: Simple greeting → routed to internal model"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Hi, how are you?"}],"max_tokens":10}'
assert_model_is_internal "Greeting → internal" "$BODY"

echo ""
echo "Test 3.2: Complex query with enterprise tier → can route to external"
if [[ -n "$GATEWAY_AUTH_URL" ]]; then
    ENT_TOKEN=$(oc create token premium-user -n vsr-demo-tier-premium --audience vsr-demo-gateway-sa --duration=1h 2>/dev/null || echo "")
    if [[ -n "$ENT_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $ENT_TOKEN" \
            -d '{"model":"auto","messages":[{"role":"user","content":"Write a comprehensive analysis of the economic implications of quantitative easing on emerging market bond yields, considering both the portfolio balance channel and the signaling channel"}],"max_tokens":10}'
        assert_status "Complex + premium" "$HEADERS" "200"
    else
        echo -e "  ${YELLOW}SKIP${RESET} Could not create enterprise token"
        ((SKIP++))
    fi
else
    echo -e "  ${YELLOW}SKIP${RESET} Auth gateway not configured"
    ((SKIP++))
fi

echo ""

# =============================================================================
# TEST GROUP 4: Department-Based Access (via Auth Gateway)
# Intern = free tier, Finance = premium tier, Principal = enterprise tier
# =============================================================================

echo "--- Group 4: Department-Based Access ---"

if [[ -n "$GATEWAY_AUTH_URL" ]]; then
    # Intern (free tier) — only internal models
    INTERN_TOKEN=$(oc create token free-user -n vsr-egress-demo --audience vsr-demo-gateway-sa --duration=1h 2>/dev/null || echo "")

    echo "Test 4.1: Intern → internal model (allowed)"
    if [[ -n "$INTERN_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $INTERN_TOKEN" \
            -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
        assert_status "Intern + internal" "$HEADERS" "200"
    fi

    echo ""
    echo "Test 4.2: Intern → external model (blocked)"
    if [[ -n "$INTERN_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $INTERN_TOKEN" \
            -d '{"model":"claude-sonnet","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
        # Should be blocked — free tier can't access claude-sonnet
        local_body=$(cat "$BODY")
        error_code=$(echo "$local_body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('code',''))" 2>/dev/null || echo "")
        if [[ "$error_code" == "403" ]]; then
            echo -e "  ${GREEN}PASS${RESET} Intern blocked from external: 403"
            ((PASS++))
        else
            echo -e "  ${RED}FAIL${RESET} Intern NOT blocked from external (expected 403, got: $error_code)"
            ((FAIL++))
        fi
    fi

    echo ""
    # Finance (premium tier) — internal + GPU + external
    FINANCE_TOKEN=$(oc create token premium-user -n vsr-demo-tier-premium --audience vsr-demo-gateway-sa --duration=1h 2>/dev/null || echo "")

    echo "Test 4.3: Finance → GPU model (allowed)"
    if [[ -n "$FINANCE_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $FINANCE_TOKEN" \
            -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"What is a bond yield?"}],"max_tokens":10}'
        assert_status "Finance + GPU" "$HEADERS" "200"
    fi

    echo ""
    echo "Test 4.4: Finance → external model (allowed for premium)"
    if [[ -n "$FINANCE_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $FINANCE_TOKEN" \
            -d '{"model":"claude-sonnet","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
        assert_status "Finance + external" "$HEADERS" "200"
    fi

    echo ""
    echo "Test 4.5: Finance + PII query → must route to internal (compliance)"
    if [[ -n "$FINANCE_TOKEN" ]]; then
        curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_AUTH_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $FINANCE_TOKEN" \
            -d '{"model":"auto","messages":[{"role":"user","content":"Check the portfolio for client John Smith, SSN 456-78-9012, and tell me the fixed income weighting"}],"max_tokens":10}'
        assert_model_is_internal "Finance + PII → internal" "$BODY"
    fi
else
    echo -e "  ${YELLOW}SKIP${RESET} Auth gateway not configured — skipping department tests"
    ((SKIP++))
fi

echo ""

# =============================================================================
# TEST GROUP 5: GPU Model Quality
# The GPU model should give coherent responses (not echo/mock)
# =============================================================================

echo "--- Group 5: GPU Model Quality ---"

echo "Test 5.1: GPU model gives real response (not echo)"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"What is Kubernetes?"}],"max_tokens":30}'
REPLY=$(python3 -c "import sys,json; d=json.load(open('$BODY')); print(d.get('choices',[{}])[0].get('message',{}).get('content','')[:80])" 2>/dev/null || echo "")
if [[ -n "$REPLY" && "$REPLY" != *"[user]"* && "$REPLY" != *"echo"* ]]; then
    echo -e "  ${GREEN}PASS${RESET} GPU real response: ${REPLY}"
    ((PASS++))
else
    echo -e "  ${RED}FAIL${RESET} GPU response looks like echo/mock: ${REPLY}"
    ((FAIL++))
fi

echo ""
echo "Test 5.2: CPU model also gives real response"
curl -sS -D "$HEADERS" -o "$BODY" -X POST "${GATEWAY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}'
assert_status "CPU model" "$HEADERS" "200"

echo ""

# ═══════════════════════════════════════════════════════════════
echo "================================================================"
echo -e "  Results: ${GREEN}${PASS} passed${RESET}, ${RED}${FAIL} failed${RESET}, ${YELLOW}${SKIP} skipped${RESET}"
echo "================================================================"
echo ""

exit $FAIL

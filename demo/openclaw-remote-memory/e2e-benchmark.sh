#!/bin/bash
# End-to-End Benchmark: Old Method vs Hybrid Memory
#
# Actually sends questions to an LLM via the proxy and checks responses.
# OLD:    bootstrap files + MEMORY.md injected as context
# HYBRID: bootstrap files + VSR search results injected as context
#
# Usage: ./e2e-benchmark.sh

set -e

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[32m'
CYAN='\033[36m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

PROXY_URL="http://127.0.0.1:11480/v1/chat/completions"
VSR_URL="http://127.0.0.1:8080"
WORKSPACE="$HOME/.openclaw/workspace"

# Bootstrap files
BOOTSTRAP_CONTENT=""
for f in "$WORKSPACE/USER.md" "$WORKSPACE/SOUL.md" "$WORKSPACE/IDENTITY.md" "$WORKSPACE/CLAUDE.md" "$WORKSPACE/AGENTS.md" "$WORKSPACE/TOOLS.md" "$WORKSPACE/HEARTBEAT.md"; do
    if [ -f "$f" ]; then
        BOOTSTRAP_CONTENT+="--- $(basename "$f") ---
$(cat "$f")

"
    fi
done

# MEMORY.md content (renamed, but same data)
MEMORY_CONTENT=""
if [ -f "$WORKSPACE/long-term-memory.md" ]; then
    MEMORY_CONTENT=$(cat "$WORKSPACE/long-term-memory.md")
fi

# VSR store ID
VS_ID=$(curl -s "$VSR_URL/v1/vector_stores" | python3 -c "
import sys, json
stores = json.load(sys.stdin).get('data') or []
for s in stores:
    if 'openclaw' in s.get('name','').lower():
        print(s['id']); break
" 2>/dev/null)

if [ -z "$VS_ID" ]; then
    echo -e "${RED}No vector store found. Is VSR running?${RESET}"
    exit 1
fi

# Verify proxy is up
if ! curl -s http://127.0.0.1:11480/health >/dev/null 2>&1; then
    echo -e "${RED}Proxy not responding on :11480${RESET}"
    exit 1
fi

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   E2E Benchmark: Old vs Hybrid — Real LLM Responses                ║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Test questions: QUERY | EXPECTED_KEYWORD | CATEGORY
TESTS=(
    "What is the user's full name?|Yossi Ovadia|IDENTITY"
    "What port does the Dude Dashboard run on?|7777|FACTUAL"
    "What is the tmux socket path used by OpenClaw?|openclaw-tmux-sockets|FACTUAL"
    "How often does the GitHub Activity Monitor cron job run?|30 min|FACTUAL"
    "What WebSocket error code was encountered when building the Live Monitor?|1006|CROSS"
    "What API endpoints does the Terminal Replay feature expose?|/api/replay|CROSS"
    "What is the LaunchAgent plist filename for OpenClaw?|ai.openclaw.gateway|CROSS"
    "What architecture mismatch bug affected bufferutil in the dashboard?|x86_64|CROSS"
)

send_to_llm() {
    local SYSTEM_PROMPT="$1"
    local QUESTION="$2"

    # Escape for JSON
    local SYS_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$SYSTEM_PROMPT")
    local Q_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$QUESTION")

    local RESPONSE=$(curl -s --max-time 120 -X POST "$PROXY_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"sonnet\",
            \"messages\": [
                {\"role\": \"system\", \"content\": $SYS_ESCAPED},
                {\"role\": \"user\", \"content\": $Q_ESCAPED}
            ]
        }" 2>/dev/null)

    echo "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d.get('choices',[{}])[0].get('message',{}).get('content','')
    print(msg)
except:
    print('ERROR: no response')
" 2>/dev/null
}

OLD_PASS=0; OLD_FAIL=0
HYBRID_PASS=0; HYBRID_FAIL=0
NUM=0
TOTAL=${#TESTS[@]}

for TEST in "${TESTS[@]}"; do
    IFS='|' read -r QUERY KEYWORD CATEGORY <<< "$TEST"
    NUM=$((NUM + 1))

    # Color category
    case "$CATEGORY" in
        IDENTITY)   CAT="${CYAN}IDENTITY${RESET}" ;;
        FACTUAL)    CAT="${GREEN}FACTUAL${RESET}" ;;
        CROSS)      CAT="${YELLOW}CROSS${RESET}" ;;
    esac

    echo -e "${BOLD}[$NUM/$TOTAL]${RESET} $CAT  ${QUERY}"
    echo ""

    # === OLD METHOD ===
    OLD_SYSTEM="You are a helpful assistant. Answer the question based ONLY on the context below. If you don't know, say 'I don't know'.

$BOOTSTRAP_CONTENT
--- MEMORY.md ---
$MEMORY_CONTENT"

    echo -ne "  ${DIM}OLD method (sending to LLM...)${RESET}"
    OLD_ANSWER=$(send_to_llm "$OLD_SYSTEM" "Answer concisely in 1-2 sentences: $QUERY")
    echo -e "\r  OLD:    $(echo "$OLD_ANSWER" | head -1 | cut -c1-90)"

    if echo "$OLD_ANSWER" | grep -qi "$KEYWORD"; then
        OLD_VERDICT="${GREEN}PASS${RESET}"
        OLD_PASS=$((OLD_PASS + 1))
    else
        OLD_VERDICT="${RED}FAIL${RESET}"
        OLD_FAIL=$((OLD_FAIL + 1))
    fi

    # === HYBRID METHOD ===
    # Get VSR search results
    VSR_RESULTS=$(curl -s -X POST "$VSR_URL/v1/vector_stores/$VS_ID/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$QUERY\", \"max_num_results\": 5}" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
parts = []
for r in data:
    parts.append(f'[{r[\"score\"]:.2f}] {r.get(\"filename\",\"?\")}:\n{r[\"content\"]}')
print('\n\n'.join(parts))
" 2>/dev/null)

    HYBRID_SYSTEM="You are a helpful assistant. Answer the question based ONLY on the context below. If you don't know, say 'I don't know'.

$BOOTSTRAP_CONTENT
--- Relevant Memory (semantic search results) ---
$VSR_RESULTS"

    echo -ne "  ${DIM}HYBRID method (sending to LLM...)${RESET}"
    HYBRID_ANSWER=$(send_to_llm "$HYBRID_SYSTEM" "Answer concisely in 1-2 sentences: $QUERY")
    echo -e "\r  HYBRID: $(echo "$HYBRID_ANSWER" | head -1 | cut -c1-90)"

    if echo "$HYBRID_ANSWER" | grep -qi "$KEYWORD"; then
        HYBRID_VERDICT="${GREEN}PASS${RESET}"
        HYBRID_PASS=$((HYBRID_PASS + 1))
    else
        HYBRID_VERDICT="${RED}FAIL${RESET}"
        HYBRID_FAIL=$((HYBRID_FAIL + 1))
    fi

    echo -e "  Result: OLD=$OLD_VERDICT  HYBRID=$HYBRID_VERDICT  (expected: $KEYWORD)"
    echo ""
done

echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo -e "${BOLD}E2E Results (real LLM responses):${RESET}"
echo ""
echo -e "  OLD    (bootstrap+MEMORY.md): ${GREEN}$OLD_PASS${RESET} pass / ${RED}$OLD_FAIL${RESET} fail out of $TOTAL"
echo -e "  HYBRID (bootstrap+VSR):       ${GREEN}$HYBRID_PASS${RESET} pass / ${RED}$HYBRID_FAIL${RESET} fail out of $TOTAL"
echo ""

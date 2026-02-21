#!/bin/bash
# Benchmark: Old Method vs Hybrid Memory
#
# Fair comparison:
#   OLD:    Bootstrap files + MEMORY.md (always loaded, every turn)
#   HYBRID: Bootstrap files (no MEMORY.md) + VSR semantic search (on demand)
#
# For each query, we check both methods and show which finds the answer.
#
# Usage: ./benchmark-hybrid-memory.sh

set -e

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[32m'
CYAN='\033[36m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

VSR_URL="http://127.0.0.1:8080"
WORKSPACE="$HOME/.openclaw/workspace"

# Bootstrap files (recognized names, always loaded by both methods)
BOOTSTRAP_FILES=(
    "$WORKSPACE/USER.md"
    "$WORKSPACE/SOUL.md"
    "$WORKSPACE/IDENTITY.md"
    "$WORKSPACE/CLAUDE.md"
    "$WORKSPACE/AGENTS.md"
    "$WORKSPACE/TOOLS.md"
    "$WORKSPACE/HEARTBEAT.md"
)

# MEMORY.md (renamed to long-term-memory.md, but we still test against it)
MEMORY_FILE="$WORKSPACE/long-term-memory.md"

# Calculate sizes
BOOTSTRAP_BYTES=0
for f in "${BOOTSTRAP_FILES[@]}"; do
    if [ -f "$f" ]; then
        BOOTSTRAP_BYTES=$((BOOTSTRAP_BYTES + $(wc -c < "$f")))
    fi
done
MEMORY_BYTES=$(wc -c < "$MEMORY_FILE" 2>/dev/null || echo "0")
OLD_TOTAL=$((BOOTSTRAP_BYTES + MEMORY_BYTES))

VS_ID=$(curl -s "$VSR_URL/v1/vector_stores" | python3 -c "
import sys, json
stores = json.load(sys.stdin).get('data') or []
for s in stores:
    if 'openclaw' in s.get('name','').lower():
        print(s['id']); break
" 2>/dev/null)

if [ -z "$VS_ID" ]; then
    echo -e "${RED}No openclaw vector store found. Is VSR running?${RESET}"
    exit 1
fi

FILE_COUNT=$(curl -s "$VSR_URL/v1/vector_stores" | python3 -c "
import sys, json
stores = json.load(sys.stdin).get('data') or []
for s in stores:
    if 'openclaw' in s.get('name','').lower():
        print(s['file_counts']['completed']); break
" 2>/dev/null || echo "?")

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   OpenClaw Memory Benchmark: Old Method vs Hybrid (Bootstrap+VSR)   ║${RESET}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}OLD method:${RESET}    bootstrap + MEMORY.md = ${OLD_TOTAL} bytes (fixed, every turn)"
echo -e "  ${BOLD}HYBRID method:${RESET} bootstrap + VSR search = ${BOOTSTRAP_BYTES}B fixed + ~1-2KB variable"
echo -e "  ${BOLD}VSR store:${RESET}     ${FILE_COUNT} files indexed (daily notes, knowledge base, docs)"
echo ""

# Test definitions: QUERY | KEYWORD | CATEGORY
# Categories:
#   IDENTITY  = in bootstrap files (USER.md, SOUL.md, etc.)
#   FACTUAL   = in MEMORY.md (now long-term-memory.md)
#   CROSS     = only in daily notes (never in MEMORY.md or bootstrap)
#   CONVENTION = in CLAUDE.md or similar
TESTS=(
    # IDENTITY: in bootstrap files
    "what is the user's name|Yossi|IDENTITY"
    "what is the user's github username|yossiovadia|IDENTITY"
    "what communication style does the user prefer|direct|IDENTITY"
    "what movie character inspires the personality|Lebowski|IDENTITY"
    "what is the assistant's emoji|🎳|IDENTITY"
    # FACTUAL: in MEMORY.md (old method has it, hybrid uses VSR)
    "what port does the dashboard run on|7777|FACTUAL"
    "what is the tmux socket path|openclaw-tmux-sockets|FACTUAL"
    "how often does github activity monitor run|30 min|FACTUAL"
    "where is the proxy source code|claude-code-proxy|FACTUAL"
    "what is the approval polling interval|15-20|FACTUAL"
    "what is the capture-pane polling rate|300ms|FACTUAL"
    # CROSS: only in daily notes — old method can NEVER find these
    "what was the websocket error when building live monitor|1006|CROSS"
    "what bug did xterm.js cause|black screen|CROSS"
    "what api endpoints does terminal replay use|/api/replay|CROSS"
    "what is the launchagent plist filename|ai.openclaw.gateway|CROSS"
    "what npm package replaced hand-rolled websocket|ws|CROSS"
    "what was the bufferutil bug|x86_64|CROSS"
    "what version of vllm in research findings|0.7.3|CROSS"
    # CONVENTION: in CLAUDE.md or workspace docs
    "how to keep claude code sessions alive|session|CONVENTION"
    "what is the workspace directory|openclaw/workspace|CONVENTION"
)

OLD_FOUND=0; OLD_MISS=0
HYBRID_FOUND=0; HYBRID_MISS=0
VSR_BYTES_TOTAL=0

printf "\n${BOLD}%-3s %-12s %-48s %-12s %-12s %-5s${RESET}\n" "#" "TYPE" "QUERY" "OLD" "HYBRID" "SCORE"
echo "──────────────────────────────────────────────────────────────────────────────────────────────"

NUM=0
for TEST in "${TESTS[@]}"; do
    IFS='|' read -r QUERY KEYWORD CATEGORY <<< "$TEST"
    NUM=$((NUM + 1))

    # === OLD METHOD: bootstrap files + MEMORY.md ===
    OLD_HIT=false
    # Check bootstrap files
    for f in "${BOOTSTRAP_FILES[@]}"; do
        if [ -f "$f" ] && grep -qi "$KEYWORD" "$f" 2>/dev/null; then
            OLD_HIT=true
            break
        fi
    done
    # Check MEMORY.md (the old method always had it)
    if ! $OLD_HIT && [ -f "$MEMORY_FILE" ] && grep -qi "$KEYWORD" "$MEMORY_FILE" 2>/dev/null; then
        OLD_HIT=true
    fi

    if $OLD_HIT; then
        OLD_RESULT="${GREEN}FOUND${RESET}"
        OLD_FOUND=$((OLD_FOUND + 1))
    else
        OLD_RESULT="${RED}MISS${RESET}"
        OLD_MISS=$((OLD_MISS + 1))
    fi

    # === HYBRID METHOD: bootstrap files + VSR search ===
    # Check bootstrap files (same as old, minus MEMORY.md)
    BOOT_HIT=false
    for f in "${BOOTSTRAP_FILES[@]}"; do
        if [ -f "$f" ] && grep -qi "$KEYWORD" "$f" 2>/dev/null; then
            BOOT_HIT=true
            break
        fi
    done

    # Check VSR semantic search
    VSR_CHECK=$(curl -s -X POST "$VSR_URL/v1/vector_stores/$VS_ID/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$QUERY\", \"max_num_results\": 5}" | python3 -c "
import sys, json, re
data = json.load(sys.stdin).get('data', [])
keyword = '$KEYWORD'
total_bytes = sum(len(r.get('content','')) for r in data)
top_score = max((r.get('score',0) for r in data), default=0)
found = any(re.search(keyword, r.get('content',''), re.IGNORECASE) for r in data)
print(f'{\"HIT\" if found else \"MISS\"}|{top_score:.2f}|{total_bytes}')
" 2>/dev/null)

    VSR_STATUS=$(echo "$VSR_CHECK" | cut -d'|' -f1)
    VSR_SCORE=$(echo "$VSR_CHECK" | cut -d'|' -f2)
    VSR_BYTES=$(echo "$VSR_CHECK" | cut -d'|' -f3)
    VSR_BYTES_TOTAL=$((VSR_BYTES_TOTAL + ${VSR_BYTES:-0}))

    VSR_HIT=false
    [ "$VSR_STATUS" = "HIT" ] && VSR_HIT=true

    # Hybrid = bootstrap OR VSR
    if $BOOT_HIT || $VSR_HIT; then
        HYBRID_RESULT="${GREEN}FOUND${RESET}"
        HYBRID_FOUND=$((HYBRID_FOUND + 1))
    else
        HYBRID_RESULT="${RED}MISS${RESET}"
        HYBRID_MISS=$((HYBRID_MISS + 1))
    fi

    # Color category
    case "$CATEGORY" in
        IDENTITY)   CAT="${CYAN}IDENTITY${RESET}" ;;
        FACTUAL)    CAT="${GREEN}FACTUAL${RESET}" ;;
        CROSS)      CAT="${YELLOW}CROSS${RESET}" ;;
        CONVENTION) CAT="${DIM}CONVENTION${RESET}" ;;
    esac

    printf "%-3s %-22b %-48s %-22b %-22b %-5s\n" \
        "$NUM" "$CAT" "$(echo "$QUERY" | cut -c1-46)" "$OLD_RESULT" "$HYBRID_RESULT" "$VSR_SCORE"
done

TOTAL=${#TESTS[@]}
HYBRID_AVG=$((BOOTSTRAP_BYTES + VSR_BYTES_TOTAL / TOTAL))

echo ""
echo "──────────────────────────────────────────────────────────────────────────────────────────────"
echo ""
echo -e "${BOLD}Results:${RESET}"
echo ""
echo -e "                    ${BOLD}OLD (bootstrap+MEMORY.md)    HYBRID (bootstrap+VSR)${RESET}"
echo -e "  Recall:           ${OLD_FOUND} / ${TOTAL}                       ${HYBRID_FOUND} / ${TOTAL}"
echo -e "  Bytes per turn:   ${OLD_TOTAL}                      ~${HYBRID_AVG}"

if [ $HYBRID_FOUND -gt $OLD_FOUND ]; then
    RECALL_DIFF=$((HYBRID_FOUND - OLD_FOUND))
    echo ""
    echo -e "  ${GREEN}Hybrid finds ${RECALL_DIFF} more answers${RESET} (cross-file: daily notes, session transcripts)"
fi

if [ $OLD_TOTAL -gt $HYBRID_AVG ]; then
    SAVINGS=$((OLD_TOTAL - HYBRID_AVG))
    PCT=$((SAVINGS * 100 / OLD_TOTAL))
    echo -e "  ${GREEN}Hybrid uses ${SAVINGS} fewer bytes/turn (${PCT}% reduction)${RESET}"
fi

echo ""
echo -e "${BOLD}Why:${RESET}"
echo -e "  OLD loads MEMORY.md every turn (${MEMORY_BYTES}B) but can't search daily notes"
echo -e "  HYBRID replaces MEMORY.md with VSR search (~1-2KB relevant chunks)"
echo -e "  VSR also searches across ${FILE_COUNT} workspace files (daily notes, docs, transcripts)"
echo ""
echo -e "${DIM}IDENTITY=USER.md/SOUL.md  FACTUAL=long-term-memory.md  CROSS=daily notes only  CONVENTION=CLAUDE.md${RESET}"
echo ""

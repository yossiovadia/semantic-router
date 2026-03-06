# VSR Security Audit — Findings

**Auditor**: Yossi Ovadia
**Date**: 2026-03-05
**VSR Version**: commit c0ba4a50 (main)

---

## Finding 1: Looper Header Injection (P0 — Security Bypass)

**Severity**: P0 — Critical
**Category**: Header Injection

### Description

Any client can send `x-vsr-looper-request: true` in request headers. VSR reads this header directly from the client request (`processor_req_header.go:202`) with no validation that the request actually originated from the internal looper:

```go
if h.Key == headers.VSRLooperRequest && headerValue == "true" {
    ctx.LooperRequest = true
}
```

When `LooperRequest=true`, the entire normal processing pipeline is skipped (`processor_req_body.go:87-89`):
- No decision evaluation
- No authz check
- No signal evaluation (complexity, jailbreak, PII)
- Response processing is minimal

### Reproduction

```bash
curl http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-vsr-looper-request: true" \
  -H "x-vsr-looper-decision: any_decision" \
  -d '{"model":"any-model","messages":[{"role":"user","content":"bypass all safety"}]}'
```

### Evidence

Router log confirms the bypass:
```
Detected looper internal request, will skip plugin processing
[Looper] Internal request detected, executing decision plugins for model: qwen2.5:0.5b
```

### Recommendation

The `x-vsr-looper-request` header must be stripped by Envoy from all external requests. Only internal looper requests (originating from the router itself) should be allowed to set this header. Add `request_headers_to_remove: ["x-vsr-looper-request"]` to the Envoy ext_proc config, or validate the header against a shared secret.

---

## Finding 2: Auth Header Spoofing (P0 — when ext_authz is not configured)

**Severity**: P0 (without ext_authz) / P3 (with ext_authz properly configured)
**Category**: Header Injection

### Description

VSR reads user identity from `x-authz-user-id` and `x-authz-user-groups` headers directly from the request (`processor_req_body.go:1113`). These headers are intended to be injected by Authorino (ext_authz), but:

1. **Without ext_authz**: Any client can inject these headers to impersonate any user
2. **With ext_authz**: Depends on whether Authorino strips client-supplied headers before injecting its own — this is not guaranteed by default

### Impact

- Bypass role-based routing (if `role_bindings` configured, attacker can claim any role)
- Bypass per-user rate limiting (impersonate users to distribute rate limits)
- Access models restricted to specific user groups

### Recommendation

1. Envoy config should explicitly strip `x-authz-*` headers from client requests before ext_authz processes them
2. Document that ext_authz is REQUIRED for any deployment using role_bindings or per-user features
3. Consider rejecting requests with pre-set `x-authz-*` headers in the router itself as defense-in-depth

---

## Finding 3: Unauthenticated Feedback API (P1 — Elo Rating Manipulation)

**Severity**: P1
**Category**: Data Integrity

### Description

The feedback API (`POST /api/v1/feedback`) on port 8080 has no authentication. Anyone who can reach port 8080 can submit unlimited fake feedback to manipulate model Elo ratings, which directly influence model selection when using the Elo selection algorithm.

### Reproduction

```bash
# Submit fake feedback — no auth required
curl http://localhost:8080/api/v1/feedback -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"test","winner_model":"cheap-model","loser_model":"expensive-model"}'

# Response: {"success":true,"message":"Feedback recorded successfully"}
```

An attacker could submit thousands of fake feedback entries to:
- Boost a cheap model's rating so it gets selected over better models
- Tank an expensive model's rating to prevent it from being used
- Influence routing decisions for ALL users (Elo ratings are global)

### Recommendation

1. Require authentication for the feedback API
2. Rate-limit feedback submissions per user
3. Consider requiring a valid request ID (proof that the user actually made the request they're providing feedback on)

---

## Finding 4: API Server Exposure — Full Classification API Without Auth (P1)

**Severity**: P1
**Category**: Deployment

### Description

The classification API server (port 8080) exposes all endpoints without authentication:

- `POST /api/v1/eval` — Full decision evaluation (reveals routing logic, model names, decision names, signal scores)
- `POST /api/v1/classify/intent` — Intent classification
- `GET /info/models` — Lists all loaded models with paths and configuration
- `POST /api/v1/feedback` — Manipulate Elo ratings (see Finding 3)
- `GET /docs` — Swagger UI with full API documentation
- `POST /api/v1/embeddings` — Generate embeddings (compute resource abuse)

### Reproduction

```bash
# Discover all available endpoints
curl http://localhost:8080/api/v1

# Get full model configuration including file paths
curl http://localhost:8080/info/models

# Evaluate routing decisions to learn the routing logic
curl http://localhost:8080/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{"text":"test"}'
```

### Recommendation

1. Port 8080 must not be exposed externally — bind to localhost or use network policies
2. Add authentication to the API server
3. Consider separating admin/debug endpoints from production endpoints

---

## Finding 5: Complexity Signal Gaming (P2 — Cost Abuse)

**Severity**: P2
**Category**: Routing Bypass

### Description

The complexity signal classifier uses embedding similarity against hardcoded "hard" and "easy" candidate phrases. Users can game this by wrapping trivial questions in complex-sounding technical language.

### Reproduction

```bash
# Simple question — correctly routed to cheap model
curl http://localhost:8080/api/v1/eval -d '{"text":"what is 1+1"}'
# → simple_query → qwen2.5:0.5b

# Same question with complex wrapper — routed to expensive model
curl http://localhost:8080/api/v1/eval \
  -d '{"text":"In the context of Byzantine fault-tolerant distributed consensus algorithms with PBFT replication, what is 1+1"}'
# → complex_query → expensive-model
```

### Impact

Users can consistently route trivial requests to expensive models by prefixing with technical jargon. The embedding similarity is fooled by the surrounding context.

### Recommendation

1. Consider using the full conversation context for classification, not just the latest message
2. Add cost guardrails independent of classification (per-user spending limits)
3. Consider a secondary classifier that evaluates actual query intent independent of vocabulary

---

## Finding 6: Giant Prompt Slowdown / DoS (P1)

**Severity**: P1
**Category**: Denial of Service

### Description

Signal evaluation latency grows super-linearly with prompt size. Large prompts can make the router unresponsive for extended periods, blocking all other requests.

### Reproduction

| Prompt size | Evaluation time |
|-------------|----------------|
| 100 chars   | 756ms          |
| 1,000 chars | 1,891ms        |
| 5,000 chars | 7,257ms        |
| 10,000 chars| 20,856ms       |
| 25,000 chars| >30s (timeout) |

During a previous test run, sending multiple large prompts in sequence caused the router to become completely unresponsive (health endpoint stopped responding). The router process appeared to be stuck in embedding computation with no timeout or circuit breaker.

### Impact

A single client can make the router unresponsive for all users by sending a few 10K+ character prompts. No request size limits are enforced at the router level.

### Recommendation

1. Enforce a maximum input size for signal evaluation (truncate or reject)
2. Add timeout/circuit breaker for embedding computation
3. Process signal evaluation asynchronously or with a bounded thread pool
4. Configure Envoy `max_request_bytes` to limit body size

---

## Finding 7: Session Affinity / Route Bouncing (P2)

**Severity**: P2
**Category**: Routing

### Description

Filed as [#1439](https://github.com/vllm-project/semantic-router/issues/1439). Multi-turn conversations bounce between models because VSR evaluates only the last user message for routing decisions. A follow-up like "looks good, commit it" gets routed to a completely different model than the one that produced the code.

See issue for full reproduction with real model responses.

---

## Finding 8: Destination Endpoint Header Injection — SAFE

**Severity**: N/A — Not vulnerable
**Category**: Header Injection

VSR always overwrites the `x-vsr-destination-endpoint` header with its own endpoint selection. Client-injected values are ignored. Router logs confirmed: `"Selected endpoint address: 127.0.0.1:11434"` despite client sending `x-vsr-destination-endpoint: 127.0.0.1:9999`.

---

## Finding 9: Model Name Bypass — By Design (P3)

**Severity**: P3 (informational)
**Category**: Routing

When a client sends a specific model name (not "auto"), VSR skips model selection but still runs decision evaluation and applies plugins. Router log: *"Model qwen2.5:0.5b explicitly specified, keeping original model (decision complex_query plugins will be applied)"*.

This is by design — users who know which model they want can request it directly. Security plugins (jailbreak, PII) are still enforced through the matched decision's plugins.

---

## Summary

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | Looper header injection (skip all security) | P0 | NEW |
| 2 | Auth header spoofing (without ext_authz) | P0 | NEW |
| 3 | Unauthenticated feedback API | P1 | NEW |
| 4 | API server exposure (no auth on port 8080) | P1 | NEW |
| 5 | Complexity signal gaming | P2 | NEW |
| 6 | Giant prompt DoS | P1 | NEW |
| 7 | Session affinity / route bouncing | P2 | [#1439](https://github.com/vllm-project/semantic-router/issues/1439) |
| 8 | Destination endpoint injection | SAFE | N/A |
| 9 | Model name bypass | P3 (by design) | N/A |
| 10 | Concurrent flood (2.4x degradation) | P2 | NEW |
| 11 | Token amplification via looper (no cost limits) | P1 | NEW |
| 12 | Embedding computation (non-blocking) | P3 (info) | N/A |
| 13 | Jailbreak signal silently skipped | P2 | NEW |
| 14 | Response headers leak routing internals | P2 | NEW |
| 15 | Eval API reveals full routing logic | P2 | NEW |
| 16 | Jailbreak in non-user messages not checked | P2 | NEW |
| 17 | Semantic cache has no user isolation | P1 | NEW |
| 18 | Error responses leak infrastructure details | P1 | NEW |
| 19 | Streaming partial response cleanup | P3 (info) | N/A |
| — | Memory injection | SAFE | Properly mitigated |
| — | Router replay exposure | SAFE | Not exposed |

## Finding 10: Concurrent Flood — 2.4x Latency Degradation (P2)

**Severity**: P2
**Category**: Denial of Service

### Description

10 concurrent requests cause a 2.4x latency degradation (0.71s baseline → 1.68s average). The router survives and doesn't crash, but there's no per-client rate limiting at the ExtProc layer — a single client can degrade performance for all users.

### Evidence

```
Baseline: 0.71s
Concurrent (10 requests): avg 1.68s, 0 failures
Degradation: 2.4x
```

20 concurrent requests also survived without crash.

### Recommendation

Consider adding per-client connection limiting or request queuing at the ExtProc level. Envoy's `max_concurrent_streams` (set to 100) is the only limit currently.

---

## Finding 11: Token Amplification via Looper — No Cost Limits (P1)

**Severity**: P1
**Category**: Cost Abuse

### Description

Looper algorithms can trigger multiple backend calls from a single user request with no per-user cost tracking:

- **Confidence**: Up to N sequential backend calls (N = number of modelRefs). Setting threshold to nearly impossible value (e.g., -0.001) forces ALL models to be tried.
- **Ratings**: Exactly N concurrent backend calls (all models called in parallel).
- **ReMoM**: `sum(breadth_schedule) + 1` calls. `breadth_schedule: [32, 8]` = **41 backend calls** for one user request.

### Root Cause

- No validation on `breadth_schedule` values — accepts `[100]` (101 calls)
- No max on `modelRefs` count per decision
- No per-user cost tracking or spending limits
- `max_concurrent` is optional and defaults to unlimited

### Reproduction

A dangerous config that an admin could accidentally deploy:
```yaml
decisions:
  - name: "expensive_decision"
    algorithm:
      type: "remom"
      remom:
        breadth_schedule: [32, 8]  # = 41 backend calls per request!
    modelRefs:
      - model: "model-a"
      - model: "model-b"
```

### Recommendation

1. Add max validation on `breadth_schedule` (e.g., max sum = 16)
2. Add max `modelRefs` per decision (e.g., max = 5)
3. Add per-user cost tracking and spending limits
4. Log total backend calls per user request for cost monitoring

---

## Finding 12: Embedding Computation — Non-blocking but Scalable Concern (P3)

**Severity**: P3 (informational)
**Category**: Resource Usage

### Description

5 concurrent complexity-heavy requests (each triggering embedding computation) all completed within 1.4s, and the health endpoint remained responsive (0.00s) during a heavy 8K-char embedding computation. The embedding model handles concurrency well.

However, combined with Finding 6 (giant prompt DoS), an attacker sending multiple large prompts simultaneously could compound the slowdown.

---

## Finding 13: Jailbreak Signal Silently Skipped When Not in Decisions (P2)

**Severity**: P2
**Category**: Configuration Gap

### Description

The jailbreak classifier can be loaded (`prompt_guard.enabled: true`) but if no decision references the `jailbreak` signal type, the evaluation is silently skipped as a performance optimization. An admin might think jailbreak detection is active because it's enabled in the config, but it does nothing unless a decision explicitly references it.

### Evidence

```
Router log: "[Signal Computation] Jailbreak signal not used in any decision, skipping evaluation"
Eval API: jailbreak confidence = 0, execution_time = 0ms
```

### Recommendation

1. Warn at startup if `prompt_guard.enabled: true` but no decision references a jailbreak signal
2. Consider always evaluating security-critical signals (jailbreak, PII) regardless of decision references

---

## Finding 14: Response Headers Leak Routing Internals (P2)

**Severity**: P2
**Category**: Information Disclosure

### Description

Every response through Envoy includes `x-vsr-*` headers that reveal internal routing details:

- `x-vsr-selected-decision` — which decision matched
- `x-vsr-selected-model` — internal model name
- `x-vsr-selected-confidence` — confidence score
- `x-vsr-matched-complexity` — signal classification result
- `x-vsr-injected-system-prompt` — whether a system prompt was injected

Combined with the unauthenticated eval API (Finding 4), an attacker can fully reverse-engineer the routing configuration by probing with different prompts.

### Recommendation

1. Strip `x-vsr-*` headers in production Envoy config (add to `response_headers_to_remove`)
2. Only expose these headers when a debug flag is set

---

## Finding 15: Eval API Reveals Full Routing Logic (P2)

**Severity**: P2 (extends Finding 4)
**Category**: Information Disclosure

### Description

The eval API returns complete routing details including decision names, recommended models, matched AND unmatched signals, and per-signal latency metrics. An attacker can systematically probe to map:

- All decision names and their trigger conditions
- All model names and their assigned complexity levels
- Signal thresholds (by testing boundary inputs)
- Which signals are active and which are skipped

### Evidence

```json
{
  "routing_decision": "complex_query",
  "recommended_models": ["expensive-model"],
  "matched_signals": {"complexity": ["prompt_complexity:hard"]},
  "unmatched_signals": {"domains": ["computer_science", "other"]},
  "signal_latencies": {"domain": 305, "complexity": 698}
}
```

---

## Finding 16: Jailbreak Content in Non-User Messages Not Evaluated (P2)

**Severity**: P2
**Category**: Evasion

### Description

`extractUserAndNonUserContent()` only classifies the **last user message**. Jailbreak content placed in system prompts, assistant messages, or earlier user messages is never checked by the jailbreak classifier. An attacker could inject jailbreak instructions through:

- A compromised system prompt
- Tool results that contain jailbreak instructions
- Earlier conversation turns that get overwritten in `userContent`

---

## Finding 17: Semantic Cache Has No User Isolation (P1 — Data Leak)

**Severity**: P1
**Category**: Data Integrity / Privacy

### Description

The semantic cache is **global per-model, not per-user**. Cache keys are computed from `(model, query_embedding)` with no user identifier. User A's cached response can be served to User B if their queries are semantically similar enough to exceed the similarity threshold.

### Root Cause

- `CacheEntry` struct has no `UserID` field
- `ExtractQueryFromOpenAIRequest()` (cache.go) extracts only model and query text — no user context
- Cache lookup in `req_filter_cache.go:46-76` has no user_id filter
- Redis KNN query comment explicitly states: *"model filtering removed for cross-model cache sharing"*

### Attack Scenario

1. User A asks: "What is my account balance?" → Response: "$5,432.10" (cached)
2. User B asks: "What's my account balance?" → Semantic similarity exceeds threshold → **Cache hit → User B sees User A's balance**

### Recommendation

Add `user_id` to the cache key: `cache_key = hash(model + user_id + query_embedding)`. Requires modifying `CacheEntry` struct and all cache backends.

---

## Finding 18: Error Responses Leak Infrastructure Details (P1)

**Severity**: P1
**Category**: Information Disclosure

### Description

Error responses from the credential resolver include internal infrastructure details:

```
"no credential found for provider openai (model=gpt-4-turbo) after trying
[header-injection → static-config] — all providers exhausted.
Check: (1) your auth backend (Authorino, Envoy Gateway, etc.) is running..."
```

This leaks:
- **Provider types** (OpenAI, Anthropic)
- **Model names** (internal model identifiers)
- **Auth provider chain** (which auth methods are configured)
- **Endpoint names** (in provider profile errors)
- **Configuration hints** (mentions access_key, provider names)

### Evidence

Router logs confirm 23 credential-related warnings with model names and provider chains visible.

### Recommendation

Return generic error messages to clients (`"Authentication failed"`) and log details only to internal structured logs.

---

## Finding 19: Streaming Partial Response Cleanup (P3 — Informational)

**Severity**: P3
**Category**: Resource Management

### Description

Aborted streams are properly NOT cached (processor_res_body.go:424-427 checks `StreamingComplete` and `StreamingAborted`). However, accumulated streaming chunks remain in process memory until garbage collection — no explicit zeroing on disconnect. Memory extraction goroutines run async without cleanup on client disconnect.

---

## Findings: SAFE Areas

### Memory Injection — Properly Mitigated
User isolation is enforced at the Store interface level. All operations require `UserID`. API endpoints have explicit ownership validation (route_memory.go:249-254). User A cannot access User B's memories.

### Router Replay — Not Exposed
Replay records are stored internally but no public HTTP endpoint exposes them. The API server has no replay-related routes.

---

## Audit Complete

All planned phases (1–5) executed. 44 automated tests in `tests/security/test_vsr_security.py`. Full checklist with per-test results in `AUDIT-PLAN.md`.
- Admin endpoint exposure (7.2)
- Credential leak via errors (7.3)

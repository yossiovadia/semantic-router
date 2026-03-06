# VSR Security & Robustness Audit Plan

**Owner**: Yossi Ovadia
**Status**: COMPLETE
**Agreed with**: Huamin
**Completed**: 2026-03-05

Goal: Document all ways a user can break, abuse, or exploit VSR. Each finding gets a severity rating, reproduction steps, and a recommendation.

**Results**: 19 findings (2 P0, 6 P1, 7 P2, 4 P3/safe). 44 automated tests in `tests/security/test_vsr_security.py`.

---

## Severity Ratings

| Rating | Meaning |
|--------|---------|
| **P0 - Critical** | Data leak, auth bypass, remote code execution |
| **P1 - High** | Cost abuse, routing bypass, denial of service |
| **P2 - Medium** | Degraded experience, incorrect routing, cache issues |
| **P3 - Low** | Edge cases, cosmetic issues, minor inconsistencies |

---

## Category 1: Routing Bypass & Gaming

### 1.1 Session Affinity / Route Bouncing
- **Status**: DONE — Finding 7 — filed as [#1439](https://github.com/vllm-project/semantic-router/issues/1439)
- **Result**: VULNERABLE (P2) — multi-turn conversations bounce between models
- **Test**: `TestSessionAffinity`

### 1.2 Model Name Bypass
- **Status**: DONE — Finding 9
- **Result**: BY DESIGN (P3) — explicit model skips selection but plugins still apply
- **Test**: `TestModelNameBypass`

### 1.3 Complexity Signal Gaming
- **Status**: DONE — Finding 5
- **Result**: VULNERABLE (P2) — jargon-wrapped trivial questions route to expensive model
- **Test**: `TestComplexityGaming`

### 1.4 Domain Signal Gaming
- **Status**: DONE
- **Result**: NOT VULNERABLE in demo config (domain classifier not fooled by keyword prefix)
- **Test**: `TestDomainSignalGaming`

### 1.5 Decision Priority Exploitation
- **Status**: COVERED by 1.3 — complexity gaming achieves the same result

---

## Category 2: Header Injection & Manipulation

### 2.1 Destination Endpoint Header Injection
- **Status**: DONE — Finding 8
- **Result**: SAFE — VSR overwrites the header
- **Test**: `TestDestinationEndpointInjection`

### 2.2 Selected Model Header Injection
- **Status**: DONE
- **Result**: SAFE — VSR ignores client-supplied x-selected-model
- **Test**: covered by `TestLooperHeaderInjection` (looper is the real bypass)

### 2.3 Auth Header Spoofing
- **Status**: DONE — Finding 2
- **Result**: VULNERABLE (P0 without ext_authz) — x-authz-user-id read from client headers
- **Test**: `TestAuthHeaderSpoofing`

### 2.4 Looper Header Injection
- **Status**: DONE — Finding 1
- **Result**: VULNERABLE (P0) — skips ALL security plugins
- **Test**: `TestLooperHeaderInjection`

---

## Category 3: Denial of Service & Resource Exhaustion

### 3.1 Giant Prompt Attack
- **Status**: DONE — Finding 6
- **Result**: VULNERABLE (P1) — 10K chars = 21s, 25K+ = timeout, router became unresponsive
- **Test**: `TestGiantPromptDoS`

### 3.2 Concurrent Request Flood
- **Status**: DONE — Finding 10
- **Result**: PARTIALLY VULNERABLE (P2) — 2.4x latency degradation, but no crash
- **Test**: `TestConcurrentFlood`

### 3.3 Embedding Model Exhaustion
- **Status**: DONE — Finding 12
- **Result**: SAFE (P3) — handles 5 concurrent evaluations well, health endpoint stays responsive
- **Test**: `TestEmbeddingExhaustion`

### 3.4 Semantic Cache Exhaustion
- **Status**: COVERED by Finding 17 — cache has no user isolation, exhaustion is secondary to poisoning

### 3.5 Memory Store Flooding
- **Status**: SAFE — memory operations require UserID, proper isolation

### 3.6 Response API Store Exhaustion
- **Status**: NOT TESTED — requires Response API enabled with persistent backend

---

## Category 4: Prompt Injection & Evasion

### 4.1 Jailbreak Detection Bypass
- **Status**: DONE — Findings 13, 16
- **Result**: VULNERABLE (P2) — jailbreak signal silently skipped when not in decisions; non-user messages not evaluated; looper header bypasses entirely
- **Test**: `TestJailbreakBypass`

### 4.2 System Prompt Extraction / Information Leakage
- **Status**: DONE — Findings 14, 15
- **Result**: VULNERABLE (P2) — x-vsr-* headers and eval API reveal full routing internals
- **Test**: `TestSystemPromptLeakage`, `TestSystemPromptExtraction`

### 4.3 Tool Call Injection
- **Status**: DONE
- **Result**: SAFE at VSR level — VSR passes tool call content as plain text, doesn't parse user messages
- **Test**: `TestToolCallInjection`

---

## Category 5: Token Theft & Cost Abuse

### 5.1 Token Amplification via Looper
- **Status**: DONE — Finding 11
- **Result**: VULNERABLE (P1) — no limits on modelRefs count or breadth_schedule values
- **Test**: `TestTokenAmplification`

### 5.2 ReMoM Cost Explosion
- **Status**: DONE — covered by Finding 11
- **Result**: VULNERABLE (P1) — breadth_schedule: [32,8] = 41 backend calls per request

### 5.3 Cache Bypass for Cost Amplification
- **Status**: COVERED by 5.1 — cost amplification via looper is more severe

### 5.4 Streaming Response Token Theft
- **Status**: DONE — Finding 19
- **Result**: MOSTLY SAFE (P3) — aborted streams not cached, minor memory cleanup gap
- **Test**: `TestStreamingCleanup`

---

## Category 6: Data & State Integrity

### 6.1 Cache Poisoning
- **Status**: DONE — Finding 17
- **Result**: VULNERABLE (P1) — cache is global per-model, not per-user
- **Test**: `TestCachePoisoning`

### 6.2 Memory Injection (MINJA)
- **Status**: DONE
- **Result**: SAFE — user isolation enforced at Store interface, API has ownership checks
- **Test**: `TestMemoryInjection`

### 6.3 Elo Rating Manipulation
- **Status**: DONE — Finding 3
- **Result**: VULNERABLE (P1) — feedback API is completely unauthenticated
- **Test**: `TestFeedbackAPIManipulation`

### 6.4 Router Replay Data Exposure
- **Status**: DONE
- **Result**: SAFE — replay API not exposed publicly
- **Test**: `TestReplayDataExposure`

---

## Category 7: Configuration & Deployment

### 7.1 API Server Exposure
- **Status**: DONE — Finding 4
- **Result**: VULNERABLE (P1) — port 8080 fully open, no auth on any endpoint
- **Test**: `TestAPIServerExposure`

### 7.2 Admin Endpoint Exposure
- **Status**: DONE
- **Result**: VULNERABLE (P1) — Envoy admin (port 19000) accessible, exposes full config
- **Test**: `TestEnvoyAdminExposure`

### 7.3 Credential Leak via Error Messages
- **Status**: DONE — Finding 18
- **Result**: VULNERABLE (P1) — errors leak model names, provider chains, endpoint names
- **Test**: `TestCredentialLeakViaErrors`

---

## Summary

| Severity | Count | Findings |
|----------|-------|----------|
| **P0** | 2 | #1 Looper header bypass, #2 Auth header spoofing |
| **P1** | 6 | #3 Feedback API, #4 API exposure, #6 Giant prompt DoS, #11 Token amplification, #17 Cache no user isolation, #18 Credential leak in errors |
| **P2** | 7 | #5 Complexity gaming, #7 Session affinity, #10 Concurrent flood, #13 Jailbreak skipped, #14 Header leakage, #15 Eval API leakage, #16 Non-user jailbreak |
| **P3/Safe** | 4 | #9 Model bypass (by design), #12 Embedding OK, #19 Streaming cleanup, + 4 safe areas |

**Test suite**: `tests/security/test_vsr_security.py` — 44 tests, all passing.
**Detailed findings**: `docs/security-audit/FINDINGS.md`

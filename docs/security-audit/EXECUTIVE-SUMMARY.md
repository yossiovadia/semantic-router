# VSR Security Audit — Executive Summary

**For**: Huamin
**From**: Yossi Ovadia
**Date**: 2026-03-06

---

## Bottom Line

VSR has **2 critical (P0)** and **6 high (P1)** vulnerabilities that must be fixed before production deployment. The P0s allow any client to bypass all security controls with a single HTTP header. The P1s include a cross-user data leak in the semantic cache and unauthenticated APIs that allow Elo rating manipulation.

All findings are backed by a reusable automated test suite (44 tests).

---

## Critical (P0) — Fix Before Any Deployment

### 1. Looper Header Bypass — All Security Skipped

Any client can send `x-vsr-looper-request: true` in the request headers. The router trusts this header unconditionally and **skips the entire security pipeline**: no decision evaluation, no jailbreak detection, no PII checks, no authz validation.

**Root cause**: `processor_req_header.go:202` reads the header from the client with no validation.

**Fix**: The router must validate that looper requests originate internally (e.g., check a shared secret, or reject looper headers from external sources). Stripping the header in Envoy config is a defense-in-depth measure but not sufficient — deployments without Envoy or with custom configs remain exposed.

### 2. Auth Header Spoofing (without ext_authz)

The router reads `x-authz-user-id` and `x-authz-user-groups` directly from client request headers. Without ext_authz (Authorino), any client can impersonate any user. This affects role-based routing, per-user rate limits, and memory isolation.

**Fix**: The router should reject requests with pre-set auth headers unless ext_authz is configured. Document that ext_authz is mandatory for any deployment using role_bindings or per-user features.

---

## High (P1) — Fix Before Production

| # | Finding | Risk | Fix Effort |
|---|---------|------|-----------|
| 3 | **Feedback API unauthenticated** — anyone can submit fake Elo feedback to shift model selection for all users | Routing manipulation | Small — add auth middleware |
| 4 | **API server (port 8080) fully open** — eval endpoint reveals all routing decisions, model names, signal scores. Model info endpoint reveals filesystem paths | Recon + routing reverse-engineering | Small — bind to localhost or add auth |
| 6 | **Giant prompt DoS** — 10K chars = 21s evaluation. 25K+ causes timeout. Multiple large prompts made the router unresponsive | Denial of service | Small — add max input size, add timeout to embedding |
| 11 | **Token amplification** — no limits on looper `breadth_schedule` or `modelRefs` count. Config `breadth_schedule: [32,8]` = 41 backend calls for 1 user request | Cost explosion | Small — add config validation |
| 17 | **Semantic cache has no user isolation** — cache key is `(model, query_embedding)` with no user_id. User A's cached response can be served to User B for semantically similar queries | **Cross-user data leak** | Medium — add user_id to cache key |
| 18 | **Error responses leak infrastructure** — auth errors include model names, endpoint names, provider chain, config hints | Recon | Small — sanitize error messages |

---

## Medium (P2) — Should Fix

- Session affinity / route bouncing ([#1439](https://github.com/vllm-project/semantic-router/issues/1439))
- Complexity signal gaming (jargon wrapping fools classifier)
- Jailbreak signal silently skipped when not referenced in decisions
- Response headers and eval API leak routing internals
- Jailbreak content in non-user messages not evaluated

---

## What's Safe

- **Destination endpoint injection**: VSR always overwrites the header (safe)
- **Memory injection**: User isolation properly enforced at Store level
- **Router replay**: Not exposed via public API
- **Streaming cleanup**: Aborted streams not cached

---

## Recommended Plan

**Immediate (this week):**
1. Fix looper header validation in Go code (P0)
2. Fix cache user isolation (P1 — data leak)
3. Sanitize error messages (P1)

**Before production:**
4. Add auth to feedback API and API server
5. Add input size limits for signal evaluation
6. Add config validation for breadth_schedule / modelRefs limits

**Architectural (next sprint):**
7. Session affinity (#1439) — wire SessionID through the pipeline

---

## Artifacts

| File | Description |
|------|-------------|
| `docs/security-audit/AUDIT-PLAN.md` | Full audit plan with all test cases and results |
| `docs/security-audit/FINDINGS.md` | Detailed findings with code references and reproduction steps |
| `tests/security/test_vsr_security.py` | 44 automated tests — run with `pytest tests/security/ -v` |
| `config/testing/config.session-affinity-demo.yaml` | Demo config used for testing |
| `config/testing/envoy.session-affinity-demo.yaml` | Demo Envoy config |

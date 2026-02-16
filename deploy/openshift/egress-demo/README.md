# vSR Egress Routing Demo

Interactive demo of vLLM Semantic Router's egress inference routing capabilities.

## What It Demonstrates

| Capability | Scenario | What Happens |
|------------|----------|-------------|
| **Body-based routing** | External Provider | Model name extracted from JSON body, routed to correct backend |
| **Multi-provider routing** | All scenarios | Single endpoint routes to OpenAI, Anthropic, and internal KServe models |
| **API translation** | API Translation | OpenAI Chat Completions format converted to Anthropic Messages API and back |
| **Tier-based access** | Free/Premium tiers | Model access restricted by subscription tier via X-MaaS-Tier header |

## Quick Start

### Local (with Ollama)

```bash
# Prerequisites: Ollama running, models downloaded (make download-models), router built (make build-router)
make run-egress-demo          # Start services
make run-egress-scenarios     # Run demo scenarios (separate terminal)

# Interactive web UI
python3 e2e/testing/demo/demo-server.py
# Open http://localhost:8888
```

### OpenShift

```bash
# Prerequisites: oc login to an OpenShift cluster
./deploy/openshift/egress-demo/deploy.sh

# Output includes the shareable Demo UI URL (HTTPS)
# Cleanup: ./deploy/openshift/egress-demo/deploy.sh --cleanup
```

**Options:**
- `--real-model` — Use Qwen3-0.6B for real inference (CPU, slower startup)
- Default: echo backend (instant responses, no model download)

## Container Image

The demo uses a custom extproc image with tier-based access control:

```
quay.io/jabadia/vsr-extproc:latest
```

**Difference from upstream image (`ghcr.io/vllm-project/semantic-router/extproc:latest`):**

The custom image includes three additions to the Go source code:
1. `ModelAccessPolicy` config type — maps tiers to allowed models (`pkg/config/config.go`)
2. `IsModelAllowedForTier()` helper — checks if a model is accessible for a tier (`pkg/config/helper.go`)
3. Tier check in `handleModelRouting()` — reads `X-MaaS-Tier` header and enforces the policy (`pkg/extproc/processor_req_body.go`)

These changes are backward-compatible: if `model_access_policy` is not in the config, access is unrestricted (existing behavior preserved).

**To rebuild:**
```bash
docker build -t quay.io/jabadia/vsr-extproc:latest -f src/vllm-sr/Dockerfile .
docker push quay.io/jabadia/vsr-extproc:latest
```

## Architecture

```
Client
  │
  ▼
Envoy (:8801) ──ExtProc──▶ vSR Router (:50051)
  │                            │
  │  ◀── routing headers ──────┘
  │     (x-selected-model, x-vsr-destination-endpoint)
  │
  ├──▶ OpenAI (External Provider)     [model: qwen2.5:1.5b]
  ├──▶ Anthropic (External Provider)  [model: claude-sonnet, API translated]
  └──▶ KServe (Internal Model)        [model: mock-llama3]
```

**Tier enforcement** happens inside vSR before routing. If `X-MaaS-Tier` header is present, vSR checks the `model_access_policy` config. In production, Authorino (via Kuadrant AuthPolicy) injects this header after consulting the MaaS API's tier resolution endpoint.

## Tracking

- **Branch:** `feat/egress-routing-demo`
- **Issue:** https://github.com/yossiovadia/semantic-router/issues/26

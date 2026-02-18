# Egress Inference Routing Demo

Interactive demo of egress inference routing with real Kubernetes authentication,
per-model tier enforcement, API translation, and external provider simulation.

## Architecture

![Phase C Deployment Architecture](architecture.png)

### Traffic Flow

1. Client sends `POST /v1/chat/completions` with a ServiceAccount token
2. **Gateway** (OpenShift Gateway API) receives the request
3. **Authorino** validates the token via KubernetesTokenReview
4. Authorino calls **MaaS API** `/v1/tiers/lookup` to resolve the user's tier from group membership
5. Authorino injects `X-MaaS-Tier` and `X-MaaS-Username` headers into the request
6. **HTTPRoute** forwards to the vSR gateway service
7. **Envoy** invokes **vSR ExtProc** via gRPC with headers + body
8. vSR checks tier policy, routes to correct backend, switches credentials, translates API format if needed

### Namespaces

| Namespace | Zone | Components |
|-----------|------|------------|
| `openshift-ingress` | Gateway | Gateway, AuthPolicy, HTTPRoute |
| `kuadrant-system` | Auth | Kuadrant operator, Authorino, Limitador |
| `vsr-egress-demo` | Internal | Envoy + vSR ExtProc, MaaS API, Demo UI, llm-katan-internal |
| `external-providers` | External | mock-anthropic (simulates Anthropic), llm-katan-external (simulates OpenAI) |
| `vsr-demo-tier-premium` | Tier | premium-user ServiceAccount |

## What It Demonstrates

| Capability | How |
|------------|-----|
| **Real Kubernetes auth** | SA tokens validated by Authorino, tier resolved via MaaS API |
| **Per-model tier enforcement** | Free tier blocked from external models, premium allowed |
| **Body-based routing** | Model name extracted from JSON body, routed to correct backend |
| **API translation** | OpenAI format converted to Anthropic Messages API and back |
| **Credential switching** | User's SA token stays internal; vSR injects provider-specific API key |
| **Egress simulation** | External providers in separate namespace, cross-namespace routing |

## Quick Start

### OpenShift

```bash
# Prerequisites: oc login to an OpenShift cluster with admin access
./deploy/openshift/egress-demo/deploy.sh

# Output includes:
#   - Demo UI URL (HTTPS)
#   - Auth Gateway URL (Kuadrant-enforced)
#   - Direct Gateway URL (no auth, for comparison)
#   - Pre-generated demo tokens (24h expiry)

# Cleanup
./deploy/openshift/egress-demo/deploy.sh --cleanup
```

**Options:**
- `--real-model` — Use Qwen3-0.6B for real inference (CPU, slower startup)
- Default: echo backend (instant responses, no model download)

### Local (with Ollama)

```bash
make run-egress-demo          # Start services
make run-egress-scenarios     # Run demo scenarios

# Interactive web UI
python3 deploy/openshift/egress-demo/demo-server.py
# Open http://localhost:8888
```

## Smoke Test

```bash
# Phase 1: Direct routing (no auth)
./deploy/openshift/egress-demo/smoke-test.sh http://<gateway-url> --phase 1

# Phase 3: Full auth flow (Kuadrant + MaaS API + tier enforcement)
./deploy/openshift/egress-demo/smoke-test.sh http://<gateway-url> --phase 3
```

## Container Image

Custom extproc image with tier-based access control:

```
quay.io/jabadia/vsr-extproc:latest
```

Adds three changes to the upstream image (`ghcr.io/vllm-project/semantic-router/extproc:latest`):
1. `ModelAccessPolicy` config type (`pkg/config/config.go`)
2. `IsModelAllowedForTier()` helper (`pkg/config/helper.go`)
3. Tier check in `handleModelRouting()` (`pkg/extproc/processor_req_body.go`)

Backward-compatible: without `model_access_policy` in config, access is unrestricted.

## Phase C Infrastructure

Phase C manifests live in `phase-c/`:

| File | What |
|------|------|
| `gatewayclass.yaml` | GatewayClass for OpenShift Gateway API |
| `gateway.yaml` | Gateway with cluster domain hostname |
| `httproute.yaml` | Routes /v1/* to vsr-gateway |
| `auth-policy.yaml` | Kuadrant AuthPolicy (token review + tier lookup + header injection) |
| `maas-api.yaml` | MaaS API deployment + RBAC + tier ConfigMap |
| `kuadrant-install.yaml` | Kuadrant operator v1.3.1 subscription |
| `kuadrant-cr.yaml` | Kuadrant CR activation |
| `demo-users.yaml` | Demo ServiceAccounts + tier namespace |
| `external-providers.yaml` | External zone namespace + mock providers |
| `serviceentry-reference.yaml` | Production Istio egress configs (reference only) |

## Tracking

- **Branch:** `feat/egress-routing-demo`
- **Issue:** https://github.com/yossiovadia/semantic-router/issues/26

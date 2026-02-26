# Egress Inference Routing Demo

Interactive demo of intelligent egress inference routing on OpenShift with real GPU model,
AI-based classification, PII detection, RAG, tier-based access control, and API translation.

**Tag:** v0.3.7 | **Branch:** `feat/egress-routing-demo` | **Issue:** [#26](https://github.com/yossiovadia/semantic-router/issues/26)

## Architecture

See [doc/architecture.md](doc/architecture.md) for full architecture details, deployment topology, and security boundaries.

![Architecture Overview](doc/architecture-overview.png)

![Deployment Topology](doc/deployment-topology.png)

## What It Demonstrates

| Capability | How |
|------------|-----|
| **Intelligent routing** | vSR classifies queries by domain (BERT LoRA), complexity (mmbert embeddings), and PII — routes to optimal model |
| **GPU inference** | Qwen2.5-7B-Instruct on NVIDIA L4 (23GB VRAM) for internal queries |
| **Complexity-based routing** | Easy queries stay internal ($0), hard+specialized go to external (Claude) |
| **PII detection & policy** | Per-destination PII policy — names/SSN allowed on-prem, email always blocked, all personal PII blocked from egress |
| **RAG (Retrieval-Augmented Generation)** | In-memory vector store with mmbert 768-dim embeddings, auto-indexed from ConfigMap on pod startup |
| **Tier-based access control** | Free/premium/enterprise tiers via K8s ServiceAccount tokens + Authorino + MaaS API |
| **API translation** | Transparent OpenAI-to-Anthropic format conversion (request + response) |
| **Rate limiting** | 10 req/min via Kuadrant Limitador |
| **Cost tracking** | Per-query cost display + running session stats (% on-prem, est. USD) |

## Demo Pages

| Page | URL Path | Purpose |
|------|----------|---------|
| Main Demo | `/` | Egress routing scenarios (direct + authenticated) |
| Financial Services | `/nb-demo` | Department-based routing, PII, RAG, complexity classification |
| Admin Console | `/admin` | User/token management, tier configuration |

## Quick Start

### Deploy on OpenShift (single command)

```bash
# Prerequisites: oc login to an OpenShift cluster with admin access
# Recommended: "RHOAI on OCP on AWS with NVIDIA GPUs" from catalog.demo.redhat.com
./deploy/openshift/egress-demo/deploy.sh

# Cleanup
./deploy/openshift/egress-demo/deploy.sh --cleanup
```

deploy.sh handles everything automatically:
- Sail Operator + Istio for Gateway API (required by Kuadrant)
- Gateway API CRDs, GatewayClass, Gateway
- Kuadrant (Authorino + Limitador) for auth + rate limiting
- GPU node detection (skips GPU setup on RHOAI clusters with pre-existing GPU)
- ML model download (domain classifier, PII detector, mmbert embeddings)
- vLLM deployment on GPU node
- RAG document indexing (auto on pod startup)
- Demo token generation (48h TTL)

### Smoke Tests

```bash
# Run sequentially (rate limit: 10 req/min shared)
./deploy/openshift/egress-demo/tests/smoke-test.sh http://<gateway-url> --phase 3
sleep 65
./deploy/openshift/egress-demo/tests/smoke-test-nb.sh http://<gateway-url>

# Expected: 30/30 main + 31/31 NB tests passing
```

## Container Images (pinned)

All images are pinned to sha256 digests for reproducible deploys. Change to `:latest` or update the digest to upgrade.

| Image | Source | Pinned Digest |
|-------|--------|---------------|
| vsr-extproc | `quay.io/jabadia/vsr-extproc` | `sha256:7e2516...` |
| vllm-openai | `vllm/vllm-openai` | `sha256:480115...` |
| maas-api | `quay.io/opendatahub/maas-api` | `sha256:7d3e65...` |
| envoy | `envoyproxy/envoy:v1.33.2` | tag-pinned |

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `vsr-egress-demo` | Internal zone — vSR router, demo UI, MaaS API, vLLM GPU |
| `external-providers` | Simulated external zone — mock Anthropic + OpenAI |
| `vsr-demo-tier-premium` | Premium tier ServiceAccount tokens |
| `vsr-demo-tier-enterprise` | Enterprise tier ServiceAccount tokens |
| `kuadrant-system` | Auth infrastructure (Authorino, Limitador, operator) |
| `openshift-ingress` | Gateway layer (OpenShift router + Istio gateway) |
| `sail-operator` | Istio control plane operator |

## Infrastructure Manifests

Platform infrastructure manifests in `infra/`:

| File | Purpose |
|------|---------|
| `gatewayclass.yaml` | GatewayClass (`istio.io/gateway-controller`) |
| `gateway.yaml` | Gateway with cluster domain hostname |
| `httproute.yaml` | Routes `/v1/*` to vsr-gateway, `/v1/tokens` + `/v1/tiers` to MaaS API |
| `auth-policy.yaml` | AuthPolicy (token review + tier lookup + X-MaaS-* header injection) |
| `ratelimit-policy.yaml` | RateLimitPolicy (10 req/min global) |
| `maas-api.yaml` | MaaS API deployment + RBAC + tier ConfigMap |
| `kuadrant-install.yaml` | Kuadrant operator v1.3.1 subscription |
| `kuadrant-cr.yaml` | Kuadrant CR activation |
| `vllm-gpu.yaml` | vLLM GPU deployment (Qwen2.5-7B on NVIDIA L4/A10G) |
| `gpu-setup.yaml` | NFD + NVIDIA GPU operator subscriptions |
| `gpu-clusterpolicy.yaml` | NVIDIA ClusterPolicy |

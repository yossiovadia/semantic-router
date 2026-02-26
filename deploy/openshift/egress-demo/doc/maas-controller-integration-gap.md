# MaaS Controller + vSR Integration Gap: Per-Subscription Token Rate Limits

## Summary

We attempted to integrate the maas-controller ([PR #427](https://github.com/opendatahub-io/models-as-a-service/pull/427)) with our vSR egress routing demo to enable per-subscription token-based rate limits (e.g., intern=50 tokens/min, finance=500, enterprise=5000). The integration failed due to an architectural mismatch between the controller's expectations and vSR's routing model.

## What We Tried

1. Installed MaaS CRDs (`MaaSSubscription`, `MaaSAuthPolicy`, `MaaSModel`) on the cluster
2. Deployed `maas-controller` (`quay.io/jrhyness/maas-controller:test`) configured for our gateway (`vsr-demo-gateway`)
3. Created `MaaSSubscription` CRDs for three tiers with different token rate limits
4. Expected the controller to generate `TokenRateLimitPolicy` resources per subscription

## What Happened

The controller logged **"model not found, cleaning up generated TRLP"** for every model reference. No `TokenRateLimitPolicy` was generated.

```
{"msg":"model not found, cleaning up generated TRLP","model":"qwen2.5-7b"}
{"msg":"model not found, cleaning up generated TRLP","model":"claude-sonnet"}
```

## Root Cause: Architectural Mismatch

The maas-controller's `MaaSSubscriptionReconciler` generates `TokenRateLimitPolicy` through a chain of resource lookups:

```
MaaSSubscription.spec.modelRefs[].name
    → looks up MaaSModel CRD by name
        → MaaSModel.spec.modelRef references LLMInferenceService
            → controller finds the per-model HTTPRoute (created by KServe/InferenceGateway)
                → generates TokenRateLimitPolicy targeting that specific HTTPRoute
```

**The controller expects per-model HTTPRoutes** — one HTTPRoute per model, each with its own path (e.g., `/llm/facebook-opt-125m-simulated/v1/chat/completions`). This is the standard KServe + InferenceGateway architecture.

**vSR uses a different architecture** — a single HTTPRoute (`/v1/*`) routing all traffic to one Envoy+vSR gateway, which internally classifies and routes to the correct backend:

```
Client → single HTTPRoute (/v1/*) → Envoy → vSR ExtProc → internal routing → backend
```

The controller has no way to attach per-subscription rate limits because there are no per-model HTTPRoutes to target.

## What Works vs What Doesn't

| Feature | MaaS Controller (KServe) | vSR Demo |
|---------|-------------------------|----------|
| Model exposure | Per-model HTTPRoute | Single gateway, internal routing |
| Rate limit target | Individual HTTPRoute per model | One shared HTTPRoute |
| Model discovery | LLMInferenceService → MaaSModel | vSR config YAML |
| Auth policy | Per-model AuthPolicy | Single gateway AuthPolicy |
| TokenRateLimitPolicy | Per-model, per-subscription | Not possible without per-model routes |

## Positive Findings

Despite the integration gap, we confirmed several things:

1. **TokenRateLimitPolicy + AuthPolicy coexist safely** — our manual AuthPolicy was untouched by the controller
2. **MaaSSubscription reconciler is independent** — creating `MaaSSubscription` without `MaaSAuthPolicy` does NOT generate unwanted AuthPolicy resources
3. **PR #427's per-subscription rate limiting works** — jrhyness's test script demonstrates different rate limits per subscription (30 vs 3000 tokens/min) working correctly in the full KServe stack

## Possible Solutions

### Option A: Add per-model HTTPRoutes to vSR demo
Create individual HTTPRoutes for each model (qwen2.5-7b, claude-sonnet, qwen2.5:1.5b) alongside the existing catch-all route. The vSR Envoy would still handle routing, but the maas-controller could attach TokenRateLimitPolicy to each model-specific route.

**Pros:** Uses the controller as designed, fully integrated
**Cons:** Changes our routing architecture, adds complexity, may conflict with vSR's internal routing

### Option B: Create TokenRateLimitPolicy manually
Skip the controller entirely. Create `TokenRateLimitPolicy` resources directly, targeting our single HTTPRoute with subscription/tier-based `when` predicates.

**Pros:** Simple, no controller dependency, works with our existing architecture
**Cons:** No CRD-based subscription management, manual policy maintenance

### Option C: Extend the maas-controller
Add support for a "centralized gateway" mode where the controller can attach rate limits to a single HTTPRoute with model-based `when` predicates derived from the request body (model field).

**Pros:** Best of both worlds — CRD automation with centralized routing
**Cons:** Requires controller code changes (PR to opendatahub-io/models-as-a-service)

### Option D: vSR as an InferenceGateway backend
Deploy vSR behind the standard InferenceGateway/KServe stack. The gateway handles per-model routing and rate limiting, vSR handles classification/PII/translation as an ExtProc.

**Pros:** Full MaaS stack compatibility, leverages all Kuadrant features
**Cons:** Significant architecture change, may duplicate routing logic

## Recommendation

For the immediate demo (next week): **Option B** — manually create TokenRateLimitPolicy. Quick, works, demonstrable.

For the product roadmap: **Option C** is the right long-term fix. The maas-controller should support centralized gateways where a single ExtProc (like vSR) handles internal routing. This is a common pattern in AI gateways and shouldn't require per-model HTTPRoutes.

## Environment

- Cluster: RHOAI on OCP on AWS with NVIDIA GPUs (catalog.demo.redhat.com)
- Kuadrant: v1.3.1 with Sail Operator v1.28.3
- vSR: quay.io/jabadia/vsr-extproc:latest
- maas-controller tested: quay.io/jrhyness/maas-controller:test (PR #427)
- Demo: https://github.com/yossiovadia/semantic-router/tree/feat/egress-routing-demo/deploy/openshift/egress-demo

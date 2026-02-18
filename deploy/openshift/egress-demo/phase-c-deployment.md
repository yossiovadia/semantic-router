# Phase C Deployment Architecture

## OpenShift Deployment Diagram

```mermaid
graph TB
    %% ── Client ──
    Client["Client<br/><i>Browser / curl</i>"]

    %% ── OpenShift Ingress Namespace ──
    subgraph ns_ingress["Namespace: openshift-ingress"]
        GW["Gateway<br/><b>vsr-demo-gateway</b><br/><i>vsr-demo.apps.cluster-*.opentlc.com</i>"]
        AP["AuthPolicy<br/><b>vsr-demo-auth-policy</b><br/><i>Enforced on Gateway</i>"]
        HR["HTTPRoute<br/><b>vsr-gateway-route</b><br/><i>/v1/* → vsr-gateway:8801</i>"]
    end

    %% ── Kuadrant System ──
    subgraph ns_kuadrant["Namespace: kuadrant-system"]
        Authorino["Authorino<br/><i>Token validation</i><br/><i>KubernetesTokenReview</i>"]
        Limitador["Limitador<br/><i>Rate limiting</i>"]
        KuadrantOp["Kuadrant Operator<br/><i>v1.3.1</i>"]
    end

    %% ── Internal Zone ──
    subgraph ns_demo["Namespace: vsr-egress-demo  (Internal Zone)"]
        subgraph pod_vsr["Pod: vsr-router"]
            Envoy["Envoy Proxy<br/><i>:8801</i>"]
            ExtProc["vSR ExtProc<br/><i>:50051 gRPC</i>"]
        end

        MaaS["MaaS API<br/><i>:8080</i><br/><b>/v1/tiers/lookup</b>"]
        TierCM["ConfigMap<br/><b>tier-to-group-mapping</b><br/><i>free / premium / enterprise</i>"]
        DemoUI["Demo UI<br/><i>:8888</i><br/><i>demo-server.py</i>"]
        TokenCM["ConfigMap<br/><b>demo-tokens</b><br/><i>pre-generated SA tokens</i>"]
        Internal["llm-katan-internal<br/><i>:8000</i><br/><i>mock-llama3</i>"]
        FreeSA["SA: free-user"]
    end

    %% ── External Zone ──
    subgraph ns_ext["Namespace: external-providers  (External Zone)"]
        ExtOpenAI["llm-katan-external<br/><i>:8000</i><br/><i>simulates OpenAI</i><br/><i>model: qwen2.5:1.5b</i>"]
        ExtAnthropic["mock-anthropic<br/><i>:8003</i><br/><i>simulates Anthropic</i><br/><i>model: claude-sonnet</i>"]
    end

    %% ── Tier Namespace ──
    subgraph ns_tier["Namespace: vsr-demo-tier-premium"]
        PremSA["SA: premium-user"]
    end

    %% ── Traffic Flow ──
    Client -->|"① POST /v1/chat/completions<br/>Authorization: Bearer &lt;SA token&gt;"| GW
    AP -.->|"attaches to"| GW
    GW -->|"② Validate token"| Authorino
    Authorino -->|"③ POST /v1/tiers/lookup<br/>{groups: [...]}"| MaaS
    MaaS -->|"④ {tier: premium}"| Authorino
    Authorino -->|"⑤ Inject headers:<br/>X-MaaS-Tier: premium<br/>X-MaaS-Username: premium-user"| GW
    GW -->|"⑥ HTTPRoute"| HR
    HR -->|"⑦ Forward to<br/>vsr-gateway:8801"| Envoy
    Envoy <-->|"⑧ gRPC ExtProc<br/>body + headers"| ExtProc
    ExtProc -->|"⑨a Route + credential switch"| Internal
    ExtProc -->|"⑨b Route + API translation<br/>+ provider API key"| ExtAnthropic
    ExtProc -->|"⑨c Route + credential switch"| ExtOpenAI

    MaaS -.->|"reads"| TierCM
    DemoUI -.->|"reads"| TokenCM
    DemoUI -->|"/auth/v1/* proxy"| GW
    DemoUI -->|"/v1/* direct"| Envoy

    %% ── Token Groups ──
    FreeSA -.->|"groups: system:serviceaccounts:vsr-egress-demo<br/>→ free tier"| MaaS
    PremSA -.->|"groups: system:serviceaccounts:vsr-demo-tier-premium<br/>→ premium tier"| MaaS

    %% ── Styling ──
    classDef gateway fill:#1a5276,stroke:#2e86c1,color:#fff
    classDef auth fill:#7d3c98,stroke:#a569bd,color:#fff
    classDef vsr fill:#1e8449,stroke:#27ae60,color:#fff
    classDef backend fill:#b9770e,stroke:#d4ac0d,color:#fff
    classDef external fill:#922b21,stroke:#e74c3c,color:#fff
    classDef config fill:#2c3e50,stroke:#7f8c8d,color:#aaa
    classDef sa fill:#1c2833,stroke:#566573,color:#bbb

    class GW,HR gateway
    class AP,Authorino,Limitador,KuadrantOp auth
    class Envoy,ExtProc,DemoUI vsr
    class Internal backend
    class ExtOpenAI,ExtAnthropic external
    class TierCM,TokenCM,MaaS config
    class FreeSA,PremSA sa
```

## Component Summary

| Component | Namespace | Purpose |
|-----------|-----------|---------|
| **Gateway** | openshift-ingress | Entry point, hostname-based routing |
| **AuthPolicy** | openshift-ingress | Token validation + tier header injection |
| **Authorino** | kuadrant-system | KubernetesTokenReview + HTTP callbacks |
| **Limitador** | kuadrant-system | Rate limiting engine |
| **MaaS API** | vsr-egress-demo | Tier lookup from SA group membership |
| **Envoy + vSR ExtProc** | vsr-egress-demo | Intelligent routing, API translation, tier enforcement |
| **Demo UI** | vsr-egress-demo | Interactive web demo + proxy |
| **llm-katan-internal** | vsr-egress-demo | Internal model (mock-llama3) |
| **llm-katan-external** | external-providers | Simulates external OpenAI-compatible API |
| **mock-anthropic** | external-providers | Simulates external Anthropic API |
| **free-user SA** | vsr-egress-demo | Free tier demo identity |
| **premium-user SA** | vsr-demo-tier-premium | Premium tier demo identity |

## Auth Flow (numbered steps from diagram)

1. Client sends request with SA token
2. Gateway delegates to Authorino for token validation
3. Authorino calls MaaS API with user's Kubernetes groups
4. MaaS API returns tier (free/premium/enterprise)
5. Authorino injects `X-MaaS-Tier` and `X-MaaS-Username` headers
6. Gateway matches HTTPRoute
7. Request forwarded to Envoy (vsr-gateway service)
8. Envoy invokes vSR ExtProc via gRPC (reads headers + body)
9. vSR checks tier policy, routes to correct backend:
   - **9a**: Internal model — direct, no translation
   - **9b**: Anthropic — credential switch + OpenAI→Anthropic translation
   - **9c**: OpenAI-compatible — credential switch, no translation

# Demo URLs - Quick Reference

**Last Updated:** 2025-10-16

## Quick Access Links

Copy and paste these URLs to access all demo components:

### Core Services

- **Envoy Gateway (HTTP):**
  `http://envoy-http-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

- **Semantic Router API:**
  `http://semantic-router-api-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

### Observability & Monitoring

- **Grafana Dashboard:**
  `https://grafana-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

- **Jaeger Tracing UI:**
  `https://jaeger-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

- **Envoy Admin:**
  `https://envoy-admin-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

### Demo Visualizations

- **Flow Visualization (Interactive):**
  `https://flow-visualization-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

- **OpenWebUI:**
  `http://openwebui-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com`

---

## Quick Commands

### Get All URLs Dynamically

```bash
# List all routes
oc get routes -n vllm-semantic-router-system

# Get specific URL
oc get route jaeger -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
```

### Check Tracing Status

```bash
./deploy/openshift/demo/toggle-tracing.sh status
```

### Run Quick Test

```bash
# Test all 10 golden prompts
./deploy/openshift/demo/curl-examples.sh all

# Or run interactive demo
python3 deploy/openshift/demo/demo-semantic-router.py
```

---

## Pre-Demo Checklist

- [ ] All pods running: `oc get pods -n vllm-semantic-router-system`
- [ ] Tracing enabled: `./deploy/openshift/demo/toggle-tracing.sh status`
- [ ] Grafana accessible and showing data
- [ ] Jaeger UI accessible (if tracing enabled)
- [ ] Flow visualization deployed and accessible
- [ ] Terminal 1 ready with log viewer
- [ ] Terminal 2 ready with demo script
- [ ] Browser tabs open for all observability tools

---

## Notes

- URLs are specific to your OpenShift cluster
- Use `oc login` before running demo scripts
- All scripts automatically discover routes dynamically
- See [DEMO-README.md](DEMO-README.md) for detailed demo flow

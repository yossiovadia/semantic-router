# Demo Scripts for Semantic Router

This directory contains demo scripts to showcase the semantic router capabilities.

## Quick Demo Guide

### 1. Live Log Viewer (Run in Terminal 1)

Shows real-time classification, routing, and security decisions:

```bash
./deploy/openshift/demo/live-semantic-router-logs.sh
```

**What it shows:**

- 📨 **Incoming requests** with user prompts
- 🛡️ **Security checks** (jailbreak detection)
- 🔍 **Classification** (category detection with confidence)
- 🎯 **Routing decisions** (which model was selected)
- 💾 **Cache hits** (semantic similarity matching)
- 🧠 **Reasoning mode** activation

**Tip:** Run this in a split terminal or separate window during your demo!

---

### 2. Interactive Demo (Run in Terminal 2)

Interactive menu-driven semantic router demo:

```bash
python3 deploy/openshift/demo/demo-semantic-router.py
```

**Features:**

1. **Single Classification** - Tests cache with same prompt (fast repeated runs)
2. **All Classifications** - Tests all 10 golden prompts
3. **Reasoning Showcase** - Chain-of-Thought vs Standard routing
4. **PII Detection Test** - Tests personal information filtering
5. **Jailbreak Detection Test** - Tests security filtering
6. **Run All Tests** - Executes all tests sequentially

**Requirements:**

- ✅ Must be logged into OpenShift (`oc login`)
- URLs are discovered automatically from routes

**What it does:**

- Goes through Envoy (same path as OpenWebUI)
- Shows routing decisions and response previews
- **Appears in Grafana dashboard!**
- Interactive - choose what to test

---

### 3. Distributed Tracing with Jaeger

Visualize the complete request flow with distributed tracing:

#### Deploy Jaeger

```bash
./deploy/openshift/demo/deploy-jaeger.sh
```

This deploys Jaeger all-in-one with:

- 📊 **Jaeger UI** for visualizing traces
- 🔗 **OTLP collector** (gRPC and HTTP)
- 💾 **In-memory storage** (demo-friendly)

#### Enable/Disable Tracing

```bash
# Enable tracing
./deploy/openshift/demo/toggle-tracing.sh enable

# Disable tracing
./deploy/openshift/demo/toggle-tracing.sh disable

# Check status
./deploy/openshift/demo/toggle-tracing.sh status
```

#### What You'll See in Jaeger

After enabling tracing and running some requests:

1. Open Jaeger UI (URL shown by toggle-tracing.sh status)
2. Select service: **vllm-semantic-router**
3. Click **Find Traces**
4. Click on a trace to see:
   - 📥 Request ingress through Envoy
   - 🔄 ExtProc classification pipeline
   - 🛡️ Security checks (jailbreak, PII)
   - 🎯 Category classification
   - 🧭 Model routing decisions
   - 💾 Cache hits/misses
   - ⏱️ End-to-end latency breakdown

**Tip:** Run some requests with `./deploy/openshift/demo/curl-examples.sh all` to generate multiple traces!

---

## Demo Flow Suggestion

### Setup (Before Demo)

```bash
# Terminal 1: Start log viewer
./deploy/openshift/demo/live-semantic-router-logs.sh

# Terminal 2: Ready to run classification test
# (don't run yet)

# Browser Tab 1: Open Grafana
# http://grafana-vllm-semantic-router-system.apps.cluster-xxx.opentlc.com

# Browser Tab 2: Open Jaeger (if tracing enabled)
# http://jaeger-vllm-semantic-router-system.apps.cluster-xxx.opentlc.com

# Browser Tab 3: Open Flow Visualization
# http://flow-visualization-vllm-semantic-router-system.apps.cluster-xxx.opentlc.com

# Browser Tab 4: Open OpenWebUI
# http://openwebui-vllm-semantic-router-system.apps.cluster-xxx.opentlc.com
```

### During Demo

1. **Show the system overview**
   - Open Flow Visualization (Browser Tab 3)
   - Click "Start Animation" to show request flow
   - Explain semantic routing concept

2. **Run interactive demo** (Terminal 2)

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   - Choose option 3 (Reasoning Showcase) to demonstrate CoT
   - Then option 2 (All Classifications)

3. **Point to live logs** (Terminal 1)
   - Show real-time classification
   - Highlight security checks (jailbreak: BENIGN)
   - Show routing decisions (Model-A vs Model-B)
   - Point out cache hits and reasoning mode activation

4. **Switch to Grafana** (Browser Tab 1)
   - Show request metrics appearing
   - Show classification category distribution
   - Show model usage breakdown

5. **Show Jaeger traces** (Browser Tab 2) - *Optional but impressive!*
   - Select service: vllm-semantic-router
   - Click "Find Traces"
   - Click on a trace to show:
     - Full request flow timeline
     - Security checks, classification, routing
     - Latency breakdown per step

6. **Show OpenWebUI integration** (Browser Tab 4)
   - Type one of the golden prompts
   - Watch it appear in logs (Terminal 1)
   - Check the trace in Jaeger (Browser Tab 2)

---

## Key Talking Points

### Classification Accuracy

- **10 golden prompts** with 100% accuracy
- Categories: Chemistry, History, Psychology, Health, Math
- Shows consistent classification behavior

### Security Features

- **Jailbreak detection** on every request
- Shows "BENIGN" for safe requests
- Confidence scores displayed

### Smart Routing

- Automatic model selection based on content
- Load balancing across Model-A and Model-B
- Routing decisions visible in logs

### Performance

- **Semantic caching** reduces latency
- Cache hits shown in logs with similarity scores
- Sub-second response times

### Observability

- Real-time logs with structured JSON
- Grafana metrics and dashboards
- **Distributed tracing** with Jaeger (OpenTelemetry)
- End-to-end request flow visualization
- Per-span latency breakdown

### Reasoning Capabilities

- **Chain-of-Thought (CoT)** for complex problems
- Enabled for math, chemistry, physics
- Standard routing for factual queries
- Automatic reasoning mode detection

---

## Troubleshooting

### Log viewer shows no output

```bash
# Check if semantic-router pod is running
oc get pods -n vllm-semantic-router-system | grep semantic-router

# Check logs manually
oc logs -n vllm-semantic-router-system deployment/semantic-router --tail=20
```

### Classification test fails

```bash
# Verify Envoy route is accessible
curl http://envoy-http-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com/v1/models

# Check if models are ready
oc get pods -n vllm-semantic-router-system
```

### Grafana doesn't show metrics

- Wait 15-30 seconds for metrics to appear
- Refresh the dashboard
- Check the time range (last 5 minutes)

---

## Cache Management

### Check Cache Status

```bash
./deploy/openshift/demo/cache-management.sh status
```

Shows recent cache activity and cached queries.

### Clear Cache (for demo)

```bash
./deploy/openshift/demo/cache-management.sh clear
```

Restarts semantic-router deployment to clear in-memory cache (~30 seconds).

### Demo Cache Feature

**Workflow to show caching in action:**

1. Clear the cache:

   ```bash
   ./deploy/openshift/demo/cache-management.sh clear
   ```

2. Run classification test (first time - no cache):

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   Choose option 2 (All Classifications)
   - Processing time: ~3-4 seconds per query
   - Logs show queries going to model

3. Run classification test again (second time - with cache):

   ```bash
   python3 deploy/openshift/demo/demo-semantic-router.py
   ```

   Choose option 2 (All Classifications) again
   - Processing time: ~400ms per query (10x faster!)
   - Logs show "💾 CACHE HIT" for all queries
   - Similarity scores ~0.99999

**Key talking point:** Cache uses **semantic similarity**, not exact string matching!

---

## Files

- `live-semantic-router-logs.sh` - Envoy traffic log viewer (security, cache, routing)
- `live-classifier-logs.sh` - Classification API log viewer
- `demo-semantic-router.py` - Interactive demo with multiple test options
- `curl-examples.sh` - Quick classification examples (direct API)
- `cache-management.sh` - Cache management helper
- `flow-visualization.html` - **Interactive flow visualization** (open in browser)
- `deploy-flow-viz.sh` - Deploy flow visualization to OpenShift
- `deploy-jaeger.sh` - Deploy Jaeger distributed tracing
- `toggle-tracing.sh` - Enable/disable tracing in semantic-router
- `CATEGORY-MODEL-MAPPING.md` - Category to model routing reference
- `demo-classification-results.json` - Test results (auto-generated)

### Flow Visualization

**Interactive visual guide** showing step-by-step request flow, security checks, classification, and routing decisions.

#### Option 1: Deploy to OpenShift (Recommended for Demos)

```bash
# Deploy as a web service with public URL
./deploy/openshift/demo/deploy-flow-viz.sh
```

This creates a lightweight nginx pod and gives you a URL like:
`http://flow-visualization-vllm-semantic-router-system.apps.cluster-xxx.opentlc.com`

Perfect for presentations - just open the URL and click "Start Animation"!

#### Option 2: Open Locally

```bash
# Open in browser (macOS)
open deploy/openshift/demo/flow-visualization.html

# Or just double-click the file
```

---

## Notes

- The log viewer uses `oc logs --follow`, so it will run indefinitely until you press Ctrl+C
- Classification test takes ~60 seconds (10 prompts with 0.5s delay between each)
- All requests go through Envoy, triggering the full routing pipeline
- Grafana metrics update in real-time (with slight delay)

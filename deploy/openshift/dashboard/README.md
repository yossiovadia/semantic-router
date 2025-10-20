# Custom Dashboard with OpenWebUI Playground for OpenShift

This directory contains the OpenShift deployment configuration and custom build for the dashboard with OpenWebUI playground integration.

## What's New

**✅ Compatible with PR #477 (HuggingChat support)**

The dashboard has been updated to work with the refactored backend (PR #477). The OpenWebUI proxy is now natively supported in the upstream codebase, so we only need to patch the frontend for OpenShift-specific hostname detection.

**New features available:**
- OpenWebUI playground at `/playground`
- **HuggingChat UI at `/huggingchat`** - Now included in deployment!
- Improved backend architecture with modular code organization

## Files

- `dashboard-deployment.yaml` - Kubernetes resources for Dashboard and ChatUI (Deployments, Services, Routes, ConfigMaps)
- `build-custom-dashboard.sh` - Builds custom dashboard image with OpenShift-specific patches
- `PlaygroundPage.tsx.patch` - Frontend patch for OpenShift hostname-aware OpenWebUI URL construction
- `README.md` - This file

## Quick Start

### Prerequisites

1. OpenShift cluster with `oc` CLI configured
2. Semantic router and OpenWebUI already deployed in `vllm-semantic-router-system` namespace
3. Docker configured to access OpenShift internal registry

### Deploy Dashboard

**Single command deployment:**

```bash
./deploy/openshift/dashboard/build-custom-dashboard.sh
```

This script automatically:

1. Creates the `dashboard-custom` imagestream if needed
2. Builds the patched dashboard image with OpenWebUI integration
3. Pushes the image to the OpenShift internal registry
4. Applies the deployment YAML if the dashboard doesn't exist, or updates the image if it does
5. Waits for the rollout to complete

### Access the Dashboard

```bash
# Get the dashboard URL
oc get route dashboard -n vllm-semantic-router-system -o jsonpath='https://{.spec.host}'
```

Navigate to:
- `/playground` - OpenWebUI playground
- `/huggingchat` - HuggingChat UI (powered by HuggingFace Chat-UI, pre-configured with 128 max_tokens)

## How It Works

### Backend (No Patching Needed!)

As of PR #477, the dashboard backend includes native support for:
- **OpenWebUI proxy** at `/embedded/openwebui/` with authorization forwarding
- **HuggingChat proxy** at `/embedded/chatui/`
- **CORS handling** for iframe embedding
- **Modular architecture** with separate packages for routing, handlers, and middleware

The backend automatically configures these proxies based on environment variables in the ConfigMap.

### Frontend Patch (OpenShift-Specific)

The `PlaygroundPage.tsx.patch` is the **only patch** needed for OpenShift. It enables the frontend to:

1. Detect when running in OpenShift (by checking the hostname)
2. Dynamically construct the correct OpenWebUI route URL
3. Load OpenWebUI in the iframe using the direct route instead of the embedded proxy path

**Before (doesn't work in OpenShift):**

```javascript
const openWebUIUrl = '/embedded/openwebui/'
```

**After (works in OpenShift):**

```javascript
const getOpenWebUIUrl = () => {
  const hostname = window.location.hostname
  if (hostname.includes('dashboard-vllm-semantic-router-system')) {
    return hostname.replace('dashboard-vllm-semantic-router-system', 'openwebui-vllm-semantic-router-system')
  }
  return '/embedded/openwebui/'
}
```

### Why Only One Patch?

**Before PR #477:** We needed to patch both backend and frontend
- Backend patch: Add OpenWebUI proxy support
- Frontend patch: OpenShift hostname detection

**After PR #477:** Only frontend patch needed!
- ✅ Backend: Native OpenWebUI/HuggingChat proxy support
- ✅ Frontend: Only needs OpenShift hostname detection

In OpenShift:
- Services are accessed via routes with unique hostnames
- The OpenWebUI URL must be dynamically constructed based on the deployment environment
- The frontend patch is applied during build time to inject this logic

## Configuration Notes

### HuggingChat Token Limits

The ChatUI deployment is pre-configured with `max_tokens=128` for both Model-A and Model-B via the `MODELS` environment variable. This ensures shorter responses optimized for chat interactions.

To change the token limit for HuggingChat, modify the `MODELS` value in the ChatUI deployment section of `dashboard-deployment.yaml`:

```yaml
- name: MODELS
  value: '[{"name":"Model-A","endpoints":[{"type":"openai"}],"parameters":{"max_tokens":128}},{"name":"Model-B","endpoints":[{"type":"openai"}],"parameters":{"max_tokens":128}}]'
```

OpenWebUI token limits are configured separately through the OpenWebUI settings interface.

## Compatibility Notes

- **✅ Compatible with PR #469** (OpenShift dashboard - your original PR)
- **✅ Compatible with PR #477** (HuggingChat support - refactored backend)
- **Build script updated** to remove obsolete `main.go.patch` reference
- **Only `PlaygroundPage.tsx.patch`** is applied during build
- Patches are maintained separately and not committed to `dashboard/`
- Only used for OpenShift demo deployment
- Original dashboard code remains untouched for upstream compatibility

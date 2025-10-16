# Custom Dashboard Build for OpenShift Demo

This directory contains patches for dashboard OpenWebUI integration, used only for the OpenShift demo deployment.

## Why Patches?

The dashboard code in `dashboard/` is owned by the upstream project maintainers. For our OpenShift demo, we need OpenWebUI integration features that are not yet in the main codebase. Instead of modifying the tracked dashboard files, we:

1. Keep the original dashboard code unchanged
2. Apply patches during custom image build for demo purposes only
3. Avoid polluting the main codebase with demo-specific changes

## Patch Files

- `main.go.patch` - Modified dashboard backend with:
  - OpenWebUI static asset proxying (`/_app/`, `/static/`, `/manifest.json`)
  - Smart API routing (sends non-Grafana APIs to OpenWebUI)
  - Authorization header forwarding for OpenWebUI authentication

- `PlaygroundPage.tsx.patch` - Modified frontend with:
  - Uses `/embedded/openwebui/` proxy path instead of `localhost:3001`

## Build Script

```bash
./deploy/openshift/demo/build-custom-dashboard.sh
```

This script:
1. Checks OpenShift registry access
2. Copies dashboard code to temporary build directory
3. Applies the patch files
4. Builds custom Docker image with patched code
5. Pushes to OpenShift internal registry
6. Updates deployment to use custom image

## Usage

The `setup-demo.sh` script can optionally call this build script if you need the latest OpenWebUI integration fixes. Otherwise, it uses the pre-built dashboard image from the main repository.

```bash
# Deploy dashboard with custom patches
./deploy/openshift/demo/build-custom-dashboard.sh

# Or use the unified setup (uses pre-built image)
./deploy/openshift/demo/setup-demo.sh
```

## Known Limitations

**OpenWebUI Iframe Embedding:**
OpenWebUI cannot fully function when embedded in an iframe from a different domain due to browser security restrictions:

- **Third-party Cookie Blocking**: Modern browsers block cookies with `SameSite=lax` in cross-origin iframes
- **Authentication Failure**: After signin, session cookies are not accessible to the iframe, causing 500 errors on `/api/v1/auths/`
- **Workaround**: Access OpenWebUI directly via its own route instead of through the dashboard

To access OpenWebUI for the demo:
```bash
# Get the direct OpenWebUI URL
oc get route openwebui -n vllm-semantic-router-system -o jsonpath='https://{.spec.host}'

# Example: https://openwebui-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com
```

This is a fundamental browser security limitation, not a bug in our implementation. The Playground page will display OpenWebUI's signin page, but authentication will fail in the iframe context.

## Notes

- Patches are maintained separately and not committed to dashboard/
- Only used for OpenShift demo deployment
- Original dashboard code remains untouched for upstream compatibility

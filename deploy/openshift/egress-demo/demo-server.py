#!/usr/bin/env python3
"""
Serves the demo web UI and proxies API requests to the gateway.
This avoids CORS issues when opening demo.html in a browser.

Usage:
    python demo-server.py [--port 8888] [--gateway http://localhost:8801]

Then open http://localhost:8888 in your browser.
"""

import argparse
import http.client
import json
import os
import re
import ssl
import time
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


GATEWAY = "http://localhost:8801"
BBR_GATEWAY = "http://localhost:8802"
AUTH_GATEWAY = ""
AUTH_GATEWAY_HOST = ""

K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
K8S_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

TIER_NAMESPACES = {
    "free": "vsr-egress-demo",
    "premium": "vsr-demo-tier-premium",
    "enterprise": "vsr-demo-tier-enterprise",
}

SYSTEM_SAS = {"default", "builder", "deployer", "demo-ui", "maas-api", "pipeline"}

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,28}[a-z0-9]$")


def k8s_request(method, path, body=None):
    """Call K8s API using mounted SA token. Returns (status_code, response_dict)."""
    try:
        with open(K8S_TOKEN_PATH) as f:
            token = f.read().strip()
    except FileNotFoundError:
        return 503, {"error": "Not running in a Kubernetes pod"}

    try:
        ctx = ssl.create_default_context(cafile=K8S_CA_PATH)
    except Exception:
        ctx = ssl._create_unverified_context()

    conn = http.client.HTTPSConnection(
        "kubernetes.default.svc", 443, context=ctx, timeout=10
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    encoded_body = json.dumps(body).encode() if body else None
    conn.request(method, path, body=encoded_body, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    try:
        return resp.status, json.loads(data)
    except json.JSONDecodeError:
        return resp.status, {"raw": data}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, directory=self.directory, **kwargs)

    def log_message(self, format, *args):
        print(f"[Demo UI] {args[0]}")

    # Only serve these files — prevent directory traversal to deploy scripts/configs
    ALLOWED_FILES = {"/demo.html", "/admin.html", "/architecture.png"}

    def _send_json(self, data, status=200):
        """Send a JSON response with CORS headers."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode())

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self):
        # Legacy token endpoint
        if self.path == "/api/tokens":
            self._serve_tokens()
            return

        # Admin API endpoints
        if self.path == "/api/admin/users":
            self._admin_get_users()
            return
        if self.path == "/api/admin/tiers":
            self._admin_get_tiers()
            return
        if self.path == "/api/admin/providers":
            self._admin_get_providers()
            return

        # Static file serving
        if self.path == "/" or self.path == "":
            self.path = "/demo.html"
        if self.path == "/admin" or self.path == "/admin.html":
            self.path = "/admin.html"
        if self.path not in self.ALLOWED_FILES:
            self.send_error(404)
            return
        super().do_GET()

    def _serve_tokens(self):
        tokens = {
            "free": os.environ.get("FREE_USER_TOKEN", ""),
            "premium": os.environ.get("PREMIUM_USER_TOKEN", ""),
        }
        self._send_json(tokens)

    def _admin_get_users(self):
        """List users (ServiceAccounts) across all tier namespaces."""
        users = []
        for tier, ns in TIER_NAMESPACES.items():
            status, data = k8s_request(
                "GET", f"/api/v1/namespaces/{ns}/serviceaccounts"
            )
            if status == 200 and "items" in data:
                for sa in data["items"]:
                    name = sa["metadata"]["name"]
                    if name not in SYSTEM_SAS:
                        users.append(
                            {
                                "name": name,
                                "tier": tier,
                                "namespace": ns,
                                "created": sa["metadata"].get(
                                    "creationTimestamp", ""
                                ),
                            }
                        )
        self._send_json(users)

    def _admin_get_tiers(self):
        """Return tier configuration (hardcoded for demo stability)."""
        tiers = [
            {
                "name": "free",
                "level": 0,
                "displayName": "Free Tier",
                "models": ["qwen3-0.6b"],
            },
            {
                "name": "premium",
                "level": 1,
                "displayName": "Premium Tier",
                "models": ["qwen3-0.6b", "qwen2.5:1.5b", "claude-sonnet"],
            },
            {
                "name": "enterprise",
                "level": 2,
                "displayName": "Enterprise Tier",
                "models": ["*"],
            },
        ]
        rate_limits = [
            {"tier": "free", "limit": "10 requests/min (Limitador)"},
            {"tier": "premium", "limit": "10 requests/min (Limitador)"},
            {"tier": "enterprise", "limit": "10 requests/min (Limitador)"},
        ]
        self._send_json({"tiers": tiers, "rateLimits": rate_limits})

    def _admin_get_providers(self):
        """Return provider/model configuration (hardcoded for demo stability)."""
        providers = [
            {
                "model": "qwen2.5:1.5b",
                "endpoint": "OpenAI External",
                "format": "openai",
                "hasKey": False,
                "keyInfo": "pass-through (no key needed)",
            },
            {
                "model": "claude-sonnet",
                "endpoint": "Anthropic External",
                "format": "anthropic",
                "hasKey": True,
                "keyPreview": "mock-ant...demo",
                "keyInfo": "Configured in vSR config (model_config.access_key)",
            },
            {
                "model": "qwen3-0.6b",
                "endpoint": "Internal (CPU)",
                "format": "openai",
                "hasKey": False,
                "keyInfo": "internal model, CPU inference (Qwen3-0.6B)",
            },
            {
                "model": "qwen2.5-7b",
                "endpoint": "Internal (GPU)",
                "format": "openai",
                "hasKey": False,
                "keyInfo": "internal model, GPU inference (Qwen2.5-7B-Instruct on A10G)",
            },
        ]
        self._send_json(providers)

    # ── POST ─────────────────────────────────────────────────────────────

    # Only proxy to known API paths — prevents SSRF to admin endpoints
    ALLOWED_API_PATHS = {"/v1/chat/completions", "/v1/models", "/v1/tokens"}

    def do_POST(self):
        # Admin API endpoints
        if self.path == "/api/admin/users":
            self._admin_create_user()
            return
        if self.path == "/api/admin/token":
            self._admin_issue_token()
            return
        if self.path == "/api/admin/test":
            self._admin_test_request()
            return

        # Gateway proxy
        target_gw = GATEWAY
        api_path = self.path

        if self.path.startswith("/auth/v1/"):
            api_path = self.path[5:]  # Strip /auth prefix
            target_gw = AUTH_GATEWAY
        elif self.path.startswith("/bbr/v1/"):
            api_path = self.path[4:]  # Strip /bbr prefix
            target_gw = BBR_GATEWAY

        if api_path not in self.ALLOWED_API_PATHS:
            self.send_error(404)
            return

        self.path = api_path
        self._proxy_to_gateway(target_gw)

    def _admin_create_user(self):
        """Create a user (ServiceAccount) in the appropriate tier namespace."""
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": f"Invalid JSON body: {e}"}, 400)
            return

        username = body.get("username", "").strip()
        tier = body.get("tier", "").strip()

        if not username or not USERNAME_RE.match(username):
            self._send_json(
                {
                    "error": "Invalid username. Must be 2-30 lowercase alphanumeric "
                    "characters or hyphens, starting and ending with alphanumeric."
                },
                400,
            )
            return

        if tier not in TIER_NAMESPACES:
            self._send_json(
                {"error": f"Invalid tier '{tier}'. Must be one of: {', '.join(TIER_NAMESPACES.keys())}"},
                400,
            )
            return

        ns = TIER_NAMESPACES[tier]
        sa_body = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": username,
                "namespace": ns,
                "labels": {
                    "app.kubernetes.io/part-of": "vsr-egress-demo",
                    "demo-role": "user",
                },
            },
        }

        status, data = k8s_request(
            "POST", f"/api/v1/namespaces/{ns}/serviceaccounts", sa_body
        )

        if status == 201:
            self._send_json(
                {
                    "name": username,
                    "tier": tier,
                    "namespace": ns,
                    "created": data.get("metadata", {}).get("creationTimestamp", ""),
                },
                201,
            )
        elif status == 409:
            self._send_json(
                {"error": f"User '{username}' already exists in {tier} tier"}, 409
            )
        else:
            self._send_json(
                {"error": f"K8s API error ({status}): {data.get('message', data)}"},
                status,
            )

    def _admin_issue_token(self):
        """Issue a token for a user via MaaS API through the Auth Gateway.

        Step 1: Get a short-lived bootstrap K8s token via TokenRequest API
        Step 2: Use that token to call MaaS /v1/tokens through the Gateway
        """
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": f"Invalid JSON body: {e}"}, 400)
            return

        username = body.get("username", "").strip()
        tier = body.get("tier", "").strip()

        if not username:
            self._send_json({"error": "username is required"}, 400)
            return
        if tier not in TIER_NAMESPACES:
            self._send_json(
                {"error": f"Invalid tier '{tier}'. Must be one of: {', '.join(TIER_NAMESPACES.keys())}"},
                400,
            )
            return

        ns = TIER_NAMESPACES[tier]
        steps = []

        # Step 1: Get bootstrap K8s token
        token_request = {
            "apiVersion": "authentication.k8s.io/v1",
            "kind": "TokenRequest",
            "spec": {
                "audiences": ["vsr-demo-gateway-sa"],
                "expirationSeconds": 600,
            },
        }

        t0 = time.time()
        status, data = k8s_request(
            "POST",
            f"/api/v1/namespaces/{ns}/serviceaccounts/{username}/token",
            token_request,
        )
        elapsed = round((time.time() - t0) * 1000)
        steps.append(
            {
                "step": 1,
                "action": "TokenRequest API",
                "detail": f"POST /api/v1/namespaces/{ns}/serviceaccounts/{username}/token",
                "status": status,
                "elapsed_ms": elapsed,
            }
        )

        if status != 201 and status != 200:
            self._send_json(
                {
                    "error": f"Failed to get bootstrap token ({status}): {data.get('message', data)}",
                    "steps": steps,
                },
                502,
            )
            return

        bootstrap_token = data.get("status", {}).get("token", "")
        if not bootstrap_token:
            self._send_json(
                {"error": "TokenRequest succeeded but no token in response", "steps": steps},
                502,
            )
            return

        steps[-1]["result"] = "Got bootstrap token (10 min TTL)"

        # Step 2: Call MaaS API through the Auth Gateway
        if not AUTH_GATEWAY:
            self._send_json(
                {
                    "error": "Auth Gateway not configured",
                    "steps": steps,
                    "bootstrapToken": bootstrap_token,
                },
                503,
            )
            return

        parsed = urllib.parse.urlparse(AUTH_GATEWAY)
        headers = {
            "Authorization": f"Bearer {bootstrap_token}",
            "Content-Type": "application/json",
        }
        if AUTH_GATEWAY_HOST:
            headers["Host"] = AUTH_GATEWAY_HOST

        try:
            t1 = time.time()
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=30
            )
            conn.request("POST", "/v1/tokens", body=b"", headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode()
            elapsed2 = round((time.time() - t1) * 1000)
            conn.close()

            steps.append(
                {
                    "step": 2,
                    "action": "MaaS API /v1/tokens (via Gateway)",
                    "detail": f"POST {AUTH_GATEWAY}/v1/tokens",
                    "status": resp.status,
                    "elapsed_ms": elapsed2,
                }
            )

            if resp.status >= 200 and resp.status < 300:
                try:
                    token_data = json.loads(resp_body)
                except json.JSONDecodeError:
                    token_data = {"raw": resp_body}
                steps[-1]["result"] = "Token issued by MaaS API"
                self._send_json(
                    {
                        "token": token_data.get("token", token_data.get("access_token", "")),
                        "expiresAt": token_data.get("expiresAt", token_data.get("expires_at", "")),
                        "issuedBy": "MaaS API",
                        "steps": steps,
                        "tokenResponse": token_data,
                    }
                )
            else:
                steps[-1]["result"] = f"MaaS API returned {resp.status}, falling back to K8s token"
                # Fall back to bootstrap token — still valid and functional
                steps.append(
                    {
                        "step": 3,
                        "action": "Fallback to K8s bootstrap token",
                        "detail": "MaaS API token issuance unavailable, using K8s TokenRequest token directly",
                        "status": 200,
                        "result": "Using bootstrap token (10 min TTL)",
                    }
                )
                self._send_json(
                    {
                        "token": bootstrap_token,
                        "expiresAt": "",
                        "issuedBy": "K8s TokenRequest (MaaS API fallback)",
                        "steps": steps,
                    },
                )

        except Exception as e:
            self._send_json(
                {
                    "error": f"Failed to reach Auth Gateway: {e}",
                    "steps": steps,
                },
                502,
            )

    def _admin_test_request(self):
        """Test a chat completion request through the Auth Gateway."""
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": f"Invalid JSON body: {e}"}, 400)
            return

        token = body.get("token", "").strip()
        model = body.get("model", "qwen3-0.6b").strip()
        message = body.get("message", "Hello").strip()

        if not token:
            self._send_json({"error": "token is required"}, 400)
            return

        if not AUTH_GATEWAY:
            self._send_json({"error": "Auth Gateway not configured"}, 503)
            return

        parsed = urllib.parse.urlparse(AUTH_GATEWAY)
        chat_body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 60,
            }
        ).encode()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if AUTH_GATEWAY_HOST:
            headers["Host"] = AUTH_GATEWAY_HOST

        try:
            t0 = time.time()
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=120
            )
            conn.request("POST", "/v1/chat/completions", body=chat_body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode()
            elapsed = round((time.time() - t0) * 1000)
            conn.close()

            # Collect response headers
            resp_headers = {}
            for key, val in resp.getheaders():
                lower = key.lower()
                if lower.startswith("x-vsr-") or lower.startswith("x-maas-") or lower.startswith("x-selected-"):
                    resp_headers[key] = val

            try:
                resp_data = json.loads(resp_body)
            except json.JSONDecodeError:
                resp_data = {"raw": resp_body[:1000]}

            self._send_json(
                {
                    "status": resp.status,
                    "elapsed_ms": elapsed,
                    "headers": resp_headers,
                    "response": resp_data,
                }
            )

        except Exception as e:
            self._send_json(
                {"error": f"Gateway request failed: {e}"}, 502
            )

    # ── DELETE ───────────────────────────────────────────────────────────

    def do_DELETE(self):
        if self.path.startswith("/api/admin/users"):
            self._admin_delete_user()
            return
        self.send_error(404)

    def _admin_delete_user(self):
        """Delete a user (ServiceAccount) from a tier namespace."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        name = params.get("name", [None])[0]
        tier = params.get("tier", [None])[0]

        if not name:
            self._send_json({"error": "Query parameter 'name' is required"}, 400)
            return
        if not tier or tier not in TIER_NAMESPACES:
            self._send_json(
                {"error": f"Invalid tier '{tier}'. Must be one of: {', '.join(TIER_NAMESPACES.keys())}"},
                400,
            )
            return

        ns = TIER_NAMESPACES[tier]
        status, data = k8s_request(
            "DELETE", f"/api/v1/namespaces/{ns}/serviceaccounts/{name}"
        )

        if status == 200 or status == 204:
            self._send_json({"deleted": name, "tier": tier, "namespace": ns})
        elif status == 404:
            self._send_json(
                {"error": f"User '{name}' not found in {tier} tier"}, 404
            )
        else:
            self._send_json(
                {"error": f"K8s API error ({status}): {data.get('message', data)}"},
                status,
            )

    # ── OPTIONS (CORS preflight) ─────────────────────────────────────────

    def do_OPTIONS(self):
        if (
            self.path.startswith("/api/admin/")
            or self.path.startswith("/auth/v1/")
            or self.path.startswith("/bbr/v1/")
            or self.path.startswith("/v1/")
            or self.path == "/api/tokens"
        ):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "content-type, x-maas-tier, Authorization",
            )
            self.end_headers()
        else:
            self.send_error(404)

    # ── Gateway proxy ────────────────────────────────────────────────────

    def _proxy_to_gateway(self, target_gw=None):
        if target_gw is None:
            target_gw = GATEWAY
        parsed = urllib.parse.urlparse(target_gw)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Forward headers (skip hop-by-hop)
        fwd_headers = {}
        for key in self.headers:
            lower = key.lower()
            if lower in ("host", "connection", "transfer-encoding"):
                continue
            fwd_headers[key] = self.headers[key]

        # For the auth gateway, set the Host header to match the Gateway listener hostname
        if target_gw == AUTH_GATEWAY and AUTH_GATEWAY_HOST:
            fwd_headers["Host"] = AUTH_GATEWAY_HOST

        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)
            conn.request("POST", self.path, body=body, headers=fwd_headers)
            resp = conn.getresponse()

            resp_body = resp.read()

            self.send_response(resp.status)
            # Forward response headers
            exposed = []
            for key, val in resp.getheaders():
                lower = key.lower()
                if lower in ("transfer-encoding", "connection"):
                    continue
                self.send_header(key, val)
                if lower.startswith("x-vsr-") or lower.startswith("x-selected-"):
                    exposed.append(key)

            # CORS headers
            self.send_header("Access-Control-Allow-Origin", "*")
            if exposed:
                self.send_header("Access-Control-Expose-Headers", ", ".join(exposed))
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()

        except Exception as e:
            error = json.dumps({"error": f"Gateway proxy error: {str(e)}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(error)


def main():
    parser = argparse.ArgumentParser(description="Demo web UI server")
    parser.add_argument("--port", type=int, default=8888, help="Port for web UI")
    parser.add_argument("--gateway", type=str, default="http://localhost:8801", help="ExtProc Gateway URL")
    parser.add_argument("--bbr-gateway", type=str, default="http://localhost:8802", help="BBR Gateway URL")
    parser.add_argument(
        "--auth-gateway",
        type=str,
        default=os.environ.get("AUTH_GATEWAY", ""),
        help="Auth Gateway URL (Gateway API endpoint)",
    )
    args = parser.parse_args()

    global GATEWAY, BBR_GATEWAY, AUTH_GATEWAY, AUTH_GATEWAY_HOST
    GATEWAY = args.gateway
    BBR_GATEWAY = args.bbr_gateway
    AUTH_GATEWAY = args.auth_gateway
    AUTH_GATEWAY_HOST = os.environ.get("AUTH_GATEWAY_HOST", "")

    server = HTTPServer(("0.0.0.0", args.port), DemoHandler)
    print(f"\n  Demo UI:      http://localhost:{args.port}")
    print(f"  Admin UI:     http://localhost:{args.port}/admin")
    print(f"  Gateway:      {GATEWAY}")
    print(f"  BBR Gateway:  {BBR_GATEWAY}")
    print(f"  Auth Gateway: {AUTH_GATEWAY or '(not set)'}")
    print(f"\n  Open http://localhost:{args.port} in your browser\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server...")


if __name__ == "__main__":
    main()

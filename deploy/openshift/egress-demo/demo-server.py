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
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


GATEWAY = "http://localhost:8801"
BBR_GATEWAY = "http://localhost:8802"
AUTH_GATEWAY = ""
AUTH_GATEWAY_HOST = ""


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, directory=self.directory, **kwargs)

    def log_message(self, format, *args):
        print(f"[Demo UI] {args[0]}")

    def do_GET(self):
        if self.path == "/api/tokens":
            self._serve_tokens()
            return
        if self.path == "/" or self.path == "":
            self.path = "/demo.html"
        super().do_GET()

    def _serve_tokens(self):
        tokens = {
            "free": os.environ.get("FREE_USER_TOKEN", ""),
            "premium": os.environ.get("PREMIUM_USER_TOKEN", ""),
        }
        body = json.dumps(tokens).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/auth/v1/"):
            self.path = self.path[5:]  # Strip /auth prefix
            self._proxy_to_gateway(AUTH_GATEWAY)
        elif self.path.startswith("/bbr/v1/"):
            self.path = self.path[4:]  # Strip /bbr prefix
            self._proxy_to_gateway(BBR_GATEWAY)
        elif self.path.startswith("/v1/"):
            self._proxy_to_gateway(GATEWAY)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        if (
            self.path.startswith("/auth/v1/")
            or self.path.startswith("/bbr/v1/")
            or self.path.startswith("/v1/")
            or self.path == "/api/tokens"
        ):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "content-type, x-maas-tier, Authorization",
            )
            self.end_headers()
        else:
            self.send_error(404)

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

#!/usr/bin/env python3
"""
Mock servers for vSR egress routing demo.

Provides two mock endpoints:
1. Mock internal model (port 8002) - responds with OpenAI-compatible format
2. Mock Anthropic endpoint (port 8003) - responds with Anthropic Messages API format

Usage:
    python mock-server.py [--internal-port 8002] [--anthropic-port 8003]
"""

import argparse
import json
import time
import uuid
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial


class MockInternalModelHandler(BaseHTTPRequestHandler):
    """Mock internal LLM model (OpenAI-compatible)"""

    def log_message(self, format, *args):
        print(f"[Mock Internal Model] {args[0]}")

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json_response({"status": "healthy"})
        elif self.path == "/v1/models":
            self.send_json_response({
                "object": "list",
                "data": [{
                    "id": "mock-llama3",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mock-internal",
                }]
            })
        else:
            self.send_json_response({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            user_content = ""
            for msg in body.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")

            response = {
                "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock-llama3"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Mock Internal Model] Response to: {user_content[:100]}",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 15,
                    "total_tokens": 40,
                },
            }
            self.send_json_response(response)
        else:
            self.send_json_response({"error": "Not found"}, 404)


class MockAnthropicHandler(BaseHTTPRequestHandler):
    """Mock Anthropic Messages API endpoint"""

    def log_message(self, format, *args):
        print(f"[Mock Anthropic] {args[0]}")

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json_response({"status": "healthy"})
        else:
            self.send_json_response({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/messages":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            # Validate Anthropic-specific headers
            api_key = self.headers.get("x-api-key", "")
            api_version = self.headers.get("anthropic-version", "")

            if not api_key:
                self.send_json_response({
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "Missing x-api-key header"},
                }, 401)
                return

            user_content = ""
            for msg in body.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                user_content = block.get("text", "")
                    else:
                        user_content = content

            # Respond in Anthropic Messages API format
            response = {
                "id": f"msg_mock_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"[Mock Anthropic] Response to: {user_content[:100]}",
                }],
                "model": body.get("model", "claude-sonnet-4-20250514"),
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 20,
                },
            }
            self.send_json_response(response)
        else:
            self.send_json_response({"error": "Not found"}, 404)


def run_server(handler_class, port, name):
    server = HTTPServer(("127.0.0.1", port), handler_class)
    print(f"[{name}] Starting on port {port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Mock servers for vSR egress demo")
    parser.add_argument("--internal-port", type=int, default=8002, help="Port for mock internal model")
    parser.add_argument("--anthropic-port", type=int, default=8003, help="Port for mock Anthropic endpoint")
    args = parser.parse_args()

    internal_thread = threading.Thread(
        target=run_server,
        args=(MockInternalModelHandler, args.internal_port, "Mock Internal Model"),
        daemon=True,
    )
    anthropic_thread = threading.Thread(
        target=run_server,
        args=(MockAnthropicHandler, args.anthropic_port, "Mock Anthropic"),
        daemon=True,
    )

    internal_thread.start()
    anthropic_thread.start()

    print(f"\nMock servers running:")
    print(f"  Internal Model: http://127.0.0.1:{args.internal_port}")
    print(f"  Mock Anthropic: http://127.0.0.1:{args.anthropic_port}")
    print(f"\nPress Ctrl+C to stop...\n")

    try:
        internal_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down mock servers...")


if __name__ == "__main__":
    main()

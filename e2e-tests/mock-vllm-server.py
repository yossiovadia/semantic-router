#!/usr/bin/env python3
"""
mock-vllm-server.py - Mock vLLM server for testing multi-model routing

This creates a simple HTTP server that mimics vLLM's OpenAI-compatible API
but responds with different model names based on the request content to simulate
intelligent routing for testing purposes.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import argparse
import threading


class MockVLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests (health check, models list)"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            models_response = {
                "object": "list",
                "data": [
                    {
                        "id": self.server.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mock-vllm",
                        "permission": [],
                        "root": self.server.model_name,
                        "parent": None
                    }
                ]
            }
            self.wfile.write(json.dumps(models_response).encode())
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; version=0.0.4; charset=utf-8')
            self.end_headers()

            # Mock Prometheus metrics that a vLLM server might expose
            mock_metrics = f"""# HELP mock_vllm_requests_total Total number of requests processed
# TYPE mock_vllm_requests_total counter
mock_vllm_requests_total{{model="{self.server.model_name}",port="{self.server.server_address[1]}"}} 42

# HELP mock_vllm_request_duration_seconds Request processing time
# TYPE mock_vllm_request_duration_seconds histogram
mock_vllm_request_duration_seconds_bucket{{model="{self.server.model_name}",le="0.1"}} 10
mock_vllm_request_duration_seconds_bucket{{model="{self.server.model_name}",le="0.5"}} 25
mock_vllm_request_duration_seconds_bucket{{model="{self.server.model_name}",le="1.0"}} 35
mock_vllm_request_duration_seconds_bucket{{model="{self.server.model_name}",le="+Inf"}} 42

# HELP mock_vllm_model_info Model information
# TYPE mock_vllm_model_info gauge
mock_vllm_model_info{{model="{self.server.model_name}",version="mock-1.0"}} 1

# HELP mock_vllm_up Server uptime
# TYPE mock_vllm_up gauge
mock_vllm_up 1
"""
            self.wfile.write(mock_metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (chat completions)"""
        # Log detailed request information for debugging
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{self.server.server_address[1]}] POST {self.path}")
        print(f"[{timestamp}] [{self.server.server_address[1]}] Headers: {dict(self.headers)}")
        
        if self.path == "/v1/chat/completions":
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))

                # Validate required fields (like real vLLM/OpenAI API)
                if "messages" not in request_data:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {
                        "error": {
                            "message": "Missing required field: 'messages'",
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": "missing_required_field"
                        }
                    }
                    self.wfile.write(json.dumps(error_response).encode())
                    return

                messages = request_data["messages"]
                if not isinstance(messages, list) or len(messages) == 0:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {
                        "error": {
                            "message": "'messages' field must be a non-empty array",
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": "invalid_messages_format"
                        }
                    }
                    self.wfile.write(json.dumps(error_response).encode())
                    return

                # Extract message content for routing simulation
                content = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        content += msg.get("content", "")
                
                # Simulate intelligent routing based on content
                selected_model = self.simulate_routing(content.lower())
                
                # Create mock response
                response = {
                    "id": f"chatcmpl-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": selected_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Mock response from {selected_model} for: {content[:50]}...",
                                "refusal": None,
                                "annotations": None,
                                "audio": None,
                                "function_call": None,
                                "tool_calls": [],
                                "reasoning_content": None
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                            "stop_reason": None,
                            "token_ids": None
                        }
                    ],
                    "service_tier": None,
                    "system_fingerprint": None,
                    "usage": {
                        "prompt_tokens": len(content.split()),
                        "total_tokens": len(content.split()) + 10,
                        "completion_tokens": 10,
                        "prompt_tokens_details": None
                    },
                    "prompt_logprobs": None,
                    "prompt_token_ids": None,
                    "kv_transfer_params": None
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": {"message": str(e), "type": "mock_error"}}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def simulate_routing(self, content):
        """Simulate intelligent model routing based on content"""
        # Math-related keywords should route to TinyLlama
        math_keywords = ["derivative", "calculate", "equation", "integral", "mathematics", "solve", "function"]
        if any(keyword in content for keyword in math_keywords):
            return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Creative writing should route to Qwen2
        creative_keywords = ["poem", "story", "creative", "write", "ocean", "sunset", "imagination"]
        if any(keyword in content for keyword in creative_keywords):
            return "Qwen/Qwen2-0.5B-Instruct"
        
        # Default to the server's primary model
        return self.server.model_name

    def log_message(self, format, *args):
        """Override to control logging"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{self.server.server_address[1]}] {format % args}")
        
        # Also log to file for debugging
        log_file = f"/home/yovadia/code/semantic-router/e2e-tests/logs/mock_{self.server.server_address[1]}.log"
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {format % args}\n")


class MockVLLMServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, model_name):
        super().__init__(server_address, RequestHandlerClass)
        self.model_name = model_name


def run_mock_server(port, model_name):
    """Run a mock vLLM server on the specified port"""
    server_address = ('127.0.0.1', port)
    httpd = MockVLLMServer(server_address, MockVLLMHandler, model_name)
    print(f"Mock vLLM server running on port {port} with model {model_name}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"Shutting down mock server on port {port}")
        httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Mock vLLM Server')
    parser.add_argument('--port', type=int, required=True, help='Port to run on')
    parser.add_argument('--model', type=str, required=True, help='Model name to serve')
    
    args = parser.parse_args()
    
    run_mock_server(args.port, args.model)


if __name__ == "__main__":
    main()
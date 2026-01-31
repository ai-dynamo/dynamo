"""Tiny CORS reverse-proxy for browser-based compliance testing.

Forwards all requests to the Dynamo backend on localhost:8000,
adding permissive CORS headers so browser-based test suites
(e.g. openresponses.org/compliance) can reach the API.

Usage:
    python scripts/cors_proxy.py [--port 8001] [--backend http://localhost:8000]
"""

import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error


class CORSProxy(BaseHTTPRequestHandler):
    backend = "http://localhost:8000"

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Expose-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _proxy(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else None

        url = f"{self.backend}{self.path}"
        req = urllib.request.Request(
            url,
            data=body,
            method=self.command,
        )
        # Forward relevant headers
        for key in ("Content-Type", "Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self._cors_headers()
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "access-control-allow-origin"):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self._cors_headers()
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(resp_body)

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy
    do_PATCH = _proxy

    def log_message(self, format, *args):
        print(f"[proxy] {self.command} {self.path} -> {args[0] if args else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend", default="http://localhost:8000")
    args = parser.parse_args()

    CORSProxy.backend = args.backend
    server = HTTPServer(("0.0.0.0", args.port), CORSProxy)
    print(f"CORS proxy listening on 0.0.0.0:{args.port} -> {args.backend}")
    server.serve_forever()

"""Tiny passthrough that strips nemo-evaluator's extra 'dataset' field before
forwarding to trtllm-serve (which has strict pydantic validation and rejects
unknown fields — vLLM accepts them, trtllm doesn't)."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlreq
from urllib.error import HTTPError

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8001"))


class Proxy(BaseHTTPRequestHandler):
    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""

        ct = self.headers.get("Content-Type", "")
        if body and "application/json" in ct:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict) and "dataset" in payload:
                    payload.pop("dataset", None)
                    body = json.dumps(payload).encode()
            except Exception:
                pass

        req = urlreq.Request(
            UPSTREAM + self.path,
            data=body if method in ("POST", "PUT", "PATCH") else None,
            method=method,
        )
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length"):
                continue
            req.add_header(k, v)
        if body:
            req.add_header("Content-Length", str(len(body)))

        try:
            with urlreq.urlopen(req, timeout=600) as r:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() in ("transfer-encoding", "content-encoding"):
                        continue
                    self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_POST(self): self._forward("POST")
    def do_GET(self): self._forward("GET")
    def do_DELETE(self): self._forward("DELETE")
    def do_PUT(self): self._forward("PUT")
    def log_message(self, fmt, *args): pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Proxy)
    print(f"strip-proxy listening on :{LISTEN_PORT} -> {UPSTREAM}", flush=True)
    srv.serve_forever()

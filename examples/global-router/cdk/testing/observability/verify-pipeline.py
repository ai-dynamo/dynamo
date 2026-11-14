#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

# Run with a generated collector configuration JSON as the first argument.
# Requires Docker Desktop; fixture traffic stays local.
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
work = pathlib.Path(tempfile.mkdtemp(prefix="gr-observability-fixture-"))


class Handler(http.server.BaseHTTPRequestHandler):
    n = 0

    def do_GET(self):
        Handler.n += 1
        n = Handler.n
        body = f"""# TYPE dynamo_frontend_active_requests gauge
dynamo_frontend_active_requests{{model="a",PodName="pod-a"}} 5
dynamo_frontend_active_requests{{model="b",PodName="pod-a"}} 7
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model="a",worker_id="w1",engine="0",PodName="pod-a"}} 3
vllm:num_requests_waiting{{model="a",worker_id="w2",engine="0",PodName="pod-a"}} 4
# TYPE dynamo_frontend_cached_tokens histogram
dynamo_frontend_cached_tokens_bucket{{le="100"}} {n}
dynamo_frontend_cached_tokens_bucket{{le="+Inf"}} {n}
dynamo_frontend_cached_tokens_sum {n*7}
dynamo_frontend_cached_tokens_count {n}
# TYPE dynamo_frontend_requests_total counter
dynamo_frontend_requests_total {[50,52,1,3][min(n-1,3)]}
# TYPE dynamo_global_frontend_requests_total counter
dynamo_global_frontend_requests_total{{status="500"}} {n}
""".encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("0.0.0.0", 18082), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
cfg["receivers"]["prometheus"]["config"]["scrape_configs"] = [
    {
        "job_name": "fixture",
        "scrape_interval": "1s",
        "static_configs": [
            {
                "targets": ["host.docker.internal:18082"],
                "labels": {
                    "ClusterName": "fixture",
                    "Component": "pool-frontend",
                    "Pool": "blue",
                    "ServingRegion": "fixture",
                },
            }
        ],
    }
]
cfg["processors"]["batch"]["timeout"] = "1s"
cfg["exporters"]["awsemf"]["output_destination"] = "stdout"
(work / "config.json").write_text(json.dumps(cfg))
cmd = [
    "docker",
    "run",
    "--rm",
    "--name",
    "gr-observability-validation",
    "-v",
    str(work / "config.json") + ":/etc/config.json:ro",
    "otel/opentelemetry-collector-contrib:0.140.1",
    "--config=/etc/config.json",
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    time.sleep(9)
    subprocess.run(
        ["docker", "stop", "-t", "2", "gr-observability-validation"],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    out = p.communicate(timeout=5)[0]
finally:
    server.shutdown()
    if p.poll() is None:
        p.kill()
(work / "pipeline.log").write_text(out)
records = []
for line in out.splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "_aws" in r:
        records.append(r)
assert records, out
cached = [
    r["dynamo_frontend_cached_tokens_sum"]
    for r in records
    if "dynamo_frontend_cached_tokens_sum" in r
]
counts = [
    r["dynamo_frontend_cached_tokens_count"]
    for r in records
    if "dynamo_frontend_cached_tokens_count" in r
]
assert cached and all(v == 7 for v in cached), cached
assert counts and all(v == 1 for v in counts), counts
errors = [
    r["dynamo_global_frontend_requests_total"]
    for r in records
    if r.get("ResponseClass") == "server_error"
]
assert errors and all(v == 1 for v in errors), errors
requests = [
    r["dynamo_frontend_requests_total"]
    for r in records
    if "dynamo_frontend_requests_total" in r
]
assert requests and all(v >= 0 for v in requests) and max(requests) <= 2, requests
active = [r for r in records if "dynamo_frontend_active_requests" in r]
assert active and {r["dynamo_frontend_active_requests"] for r in active} == {
    5,
    7,
}, active
waiting = [r for r in records if "vllm:num_requests_waiting" in r]
assert waiting and {r["vllm:num_requests_waiting"] for r in waiting} == {3, 4}, waiting
for r in waiting:
    assert any(
        "worker_id" in d and "engine" in d and "PodName" in d
        for m in r["_aws"]["CloudWatchMetrics"]
        for d in m["Dimensions"]
    ), r
print(
    json.dumps(
        {
            "scrapes": Handler.n,
            "emfRecords": len(records),
            "cachedTokenDeltas": cached,
            "requestCountDeltas": counts,
            "result": "passed",
            "evidence": str(work),
        }
    )
)

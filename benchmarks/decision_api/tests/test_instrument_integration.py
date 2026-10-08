# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only wire replay through the real pinned AIPerf process."""

import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from dynamo_decision_perf.runner import RunSpec, execute, freeze_run
from test_contracts import fixture, native_fixture

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.timeout(120),
]


@pytest.mark.parametrize("dialect", ["oai", "systemone", "sglang_native"])
@pytest.mark.skipif(
    os.environ.get("DECISION_PERF_INTEGRATION") != "1",
    reason="set DECISION_PERF_INTEGRATION=1 for real AIPerf subprocess validation",
)
def test_aiperf_exact_replay_known_delay_and_bad_success(tmp_path, dialect):
    request, response = fixture(dialect)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(
                (body, self.headers.get("X-Decision-Measurement-ID"), self.path)
            )
            time.sleep(0.05)
            index = len(seen)
            content = json.dumps({} if index == 2 else response).encode()
            self.send_response(429 if index == 3 else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header(
                "X-Request-ID", "server-" + self.headers["X-Decision-Measurement-ID"]
            )
            self.send_header(
                "X-TypeSafe-Request-ID",
                "server-" + self.headers["X-Decision-Measurement-ID"],
            )
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "input.json"
    dataset.write_text(
        json.dumps({"data": [{"session_id": "cpu-fixture", "payloads": [request]}]})
    )
    spec = RunSpec(
        dialect,
        f"http://127.0.0.1:{server.server_port}",
        "m",
        requests=4,
        duration=15,
        timeout=5,
    )
    run = tmp_path / "run"
    freeze_run(
        run, spec, dataset, {"kind": "cpu_instrument", "series_id": "instrument"}
    )
    try:
        assert execute(run, spec, shutil.which("aiperf")) == 0, (
            run / "aiperf.log"
        ).read_text()[-12000:]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(seen) == 4, "no retries may hide overload or change denominators"
    assert all(body == request for body, _, _ in seen)
    assert len({rid for _, rid, _ in seen}) == 4
    assert {path for _, _, path in seen} == {
        "/v1/systemone" if dialect == "systemone" else "/v1/decisions"
    }
    raw = [
        json.loads(line)
        for line in (run / "aiperf" / "profile_export_raw.jsonl")
        .read_text()
        .splitlines()
    ]
    records = [
        json.loads(line)
        for line in (run / "aiperf" / "profile_export.jsonl").read_text().splitlines()
    ]
    assert len(raw) == len(records) == 4
    assert sum(bool(row.get("error")) for row in records) == 2
    for row in records:
        metrics = row.get("metrics", {})
        assert not any(
            key in metrics
            for key in (
                "time_to_first_token",
                "inter_token_latency",
                "output_token_throughput",
            )
        )
        if not row.get("error"):
            assert metrics["request_latency"]["value"] >= 45


@pytest.mark.skipif(
    os.environ.get("DECISION_PERF_INTEGRATION") != "1",
    reason="set DECISION_PERF_INTEGRATION=1 for real SSE instrumentation",
)
def test_native_sse_terminal_score_and_per_send_salt(tmp_path):
    request, response = native_fixture()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            intermediate = {"output_ids": [], "meta_info": {"completion_tokens": 0}}
            content = (
                "data: "
                + json.dumps(intermediate)
                + "\n\ndata: "
                + json.dumps(response)
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "input.json"
    dataset.write_text(
        json.dumps({"data": [{"session_id": "native-fixture", "payloads": [request]}]})
    )
    spec = RunSpec(
        "native_score",
        f"http://127.0.0.1:{server.server_port}",
        "m",
        requests=2,
        duration=15,
        timeout=5,
    )
    run = tmp_path / "run"
    freeze_run(run, spec, dataset, {"kind": "cpu_instrument"})
    try:
        assert execute(run, spec, shutil.which("aiperf")) == 0, (
            run / "aiperf.log"
        ).read_text()[-10000:]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(seen) == 2 and seen[0]["cache_salt"] != seen[1]["cache_salt"]
    assert all(
        {key: value for key, value in body.items() if key != "cache_salt"} == request
        for body in seen
    )
    records = [
        json.loads(line)
        for line in (run / "aiperf" / "profile_export.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2 and not any(row.get("error") for row in records)
    assert all("time_to_first_token" not in row.get("metrics", {}) for row in records)

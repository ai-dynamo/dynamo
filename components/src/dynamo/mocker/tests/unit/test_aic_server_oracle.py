# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import dynamo._internal.aic as aic
from dynamo._internal.aic import ServerOracleSession, create_session

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


class _ServerOracleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append((self.path, payload))

        if self.path in self.server.responses:
            body = self.server.responses[self.path]
        elif self.path == "/v1/mocker/predict_decode":
            body = {"latency_ms": 8.125}
        elif self.path == "/v1/mocker/predict_prefill":
            body = {"latency_us": 2500.0}
        elif self.path == "/v1/mocker/estimate_num_gpu_blocks":
            body = {"num_gpu_blocks": 12345}
        else:
            self.send_response(404)
            self.end_headers()
            return

        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


@pytest.fixture
def server_oracle():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ServerOracleHandler)
    server.requests = []
    server.responses = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_create_session_supports_server_oracle_backend(server_oracle):
    base_url, server = server_oracle

    session = create_session(
        backend_name="server_oracle",
        system=base_url,
        model_path="/models/minimax",
        tp_size=4,
    )

    assert session.predict_decode(batch_size=4, isl=8192, osl=2) == pytest.approx(8.125)
    assert session.predict_prefill(
        batch_size=1, effective_isl=128, prefix=0
    ) == pytest.approx(2.5)
    assert server.requests == [
        (
            "/v1/mocker/predict_decode",
            {"batch_size": 4, "context_length": 8192},
        ),
        (
            "/v1/mocker/predict_prefill",
            {"batch_size": 1, "effective_isl": 128, "prefix": 0},
        ),
    ]


def test_server_oracle_session_uses_env_url_when_system_is_not_url(
    monkeypatch, server_oracle
):
    base_url, _ = server_oracle
    monkeypatch.setenv("DYNAMO_SERVER_ORACLE_URL", base_url)

    session = ServerOracleSession("h200_sxm", timeout_s=1.0)

    assert session.base_url == base_url
    assert session.predict_decode(batch_size=2, isl=1024, osl=2) == pytest.approx(8.125)


@pytest.mark.parametrize("url", ["file:///tmp/oracle", "localhost:8010"])
def test_server_oracle_session_rejects_invalid_explicit_url(url):
    with pytest.raises(ValueError, match="http:// or https://"):
        ServerOracleSession(url, timeout_s=1.0)


def test_server_oracle_session_rejects_invalid_env_url(monkeypatch):
    monkeypatch.setenv("DYNAMO_SERVER_ORACLE_URL", "file:///tmp/oracle")

    with pytest.raises(ValueError, match="http:// or https://"):
        ServerOracleSession("h200_sxm", timeout_s=1.0)


@pytest.mark.parametrize(
    "response_body",
    [
        {"latency_ms": "nan"},
        {"latency_ms": "inf"},
        {"latency_us": -1.0},
    ],
)
def test_server_oracle_session_rejects_invalid_latency(response_body, server_oracle):
    base_url, server = server_oracle
    server.responses["/v1/mocker/predict_decode"] = response_body
    session = ServerOracleSession(base_url, timeout_s=1.0)

    with pytest.raises(RuntimeError, match="invalid latency_ms"):
        session.predict_decode(batch_size=2, isl=1024, osl=2)


def test_server_oracle_session_caches_latency_predictions(server_oracle):
    base_url, server = server_oracle
    session = ServerOracleSession(base_url, timeout_s=1.0)

    assert session.predict_decode(batch_size=2, isl=1024, osl=2) == pytest.approx(8.125)
    assert session.predict_decode(batch_size=2, isl=1024, osl=2) == pytest.approx(8.125)
    assert session.predict_decode(batch_size=2, isl=1025, osl=2) == pytest.approx(8.125)
    assert session.predict_prefill(
        batch_size=1, effective_isl=128, prefix=0
    ) == pytest.approx(2.5)
    assert session.predict_prefill(
        batch_size=1, effective_isl=128, prefix=0
    ) == pytest.approx(2.5)

    assert server.requests == [
        (
            "/v1/mocker/predict_decode",
            {"batch_size": 2, "context_length": 1024},
        ),
        (
            "/v1/mocker/predict_decode",
            {"batch_size": 2, "context_length": 1025},
        ),
        (
            "/v1/mocker/predict_prefill",
            {"batch_size": 1, "effective_isl": 128, "prefix": 0},
        ),
    ]


def test_server_oracle_session_estimates_num_gpu_blocks(server_oracle):
    base_url, server = server_oracle

    session = ServerOracleSession(base_url, timeout_s=1.0)

    assert (
        session.estimate_num_gpu_blocks(
            model_path="/models/minimax",
            tp_size=4,
            block_size=64,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.8,
            mem_fraction_static=None,
            free_gpu_memory_fraction=0.9,
            backend_version="mock-version",
            moe_tp_size=2,
            moe_ep_size=1,
            attention_dp_size=1,
            engine_type="vllm",
        )
        == 12345
    )
    assert server.requests == [
        (
            "/v1/mocker/estimate_num_gpu_blocks",
            {
                "model_path": "/models/minimax",
                "tp_size": 4,
                "block_size": 64,
                "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.8,
                "free_gpu_memory_fraction": 0.9,
                "backend_version": "mock-version",
                "moe_tp_size": 2,
                "moe_ep_size": 1,
                "attention_dp_size": 1,
                "engine_type": "vllm",
            },
        )
    ]


@pytest.mark.parametrize("num_gpu_blocks", [123.9, "123", True])
def test_server_oracle_session_rejects_invalid_num_gpu_blocks(
    num_gpu_blocks, server_oracle
):
    base_url, server = server_oracle
    server.responses["/v1/mocker/estimate_num_gpu_blocks"] = {
        "num_gpu_blocks": num_gpu_blocks
    }
    session = ServerOracleSession(base_url, timeout_s=1.0)

    with pytest.raises(RuntimeError, match="invalid num_gpu_blocks"):
        session.estimate_num_gpu_blocks(
            model_path="/models/minimax",
            tp_size=1,
            block_size=64,
            max_num_batched_tokens=4096,
        )


def test_estimate_num_gpu_blocks_routes_server_oracle_without_loading_aic(
    monkeypatch, server_oracle
):
    base_url, server = server_oracle

    def fail_load_aic():
        raise AssertionError("server_oracle capacity must not load AIC")

    monkeypatch.setattr(aic, "_load_aiconfigurator", fail_load_aic)

    assert (
        aic.estimate_num_gpu_blocks(
            backend_name="server_oracle",
            system=base_url,
            model_path="/models/minimax",
            tp_size=4,
            block_size=64,
            max_num_batched_tokens=4096,
            engine_type="vllm",
        )
        == 12345
    )
    assert server.requests == [
        (
            "/v1/mocker/estimate_num_gpu_blocks",
            {
                "model_path": "/models/minimax",
                "tp_size": 4,
                "block_size": 64,
                "max_num_batched_tokens": 4096,
                "gpu_memory_utilization": 0.9,
                "engine_type": "vllm",
            },
        )
    ]

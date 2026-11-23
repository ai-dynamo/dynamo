# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgpack
import pytest
import zmq
import zmq.asyncio

from dynamo.llm import (
    KvRouter,
    KvRouterConfig,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    MooncakeClient,
    MooncakeCommandError,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.integration]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """This control path uses file discovery and ZMQ events."""
    yield


@pytest.fixture
def admin_server():
    requests = []
    failures = set()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.connection.settimeout(2)
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if len(requests) in failures:
                self.send_error(503)
                return
            response = json.dumps(
                {
                    "results": [
                        {"key": key, "error_code": -704 if key.endswith("_v") else 0}
                        for key in payload["keys"]
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests, failures
        finally:
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_resolves_events_and_preserves_python_results(
    temp_file_store, monkeypatch, admin_server
):
    admin_url, requests, failures = admin_server
    monkeypatch.setenv("DYN_EVENT_PLANE", "zmq")
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
    context = zmq.asyncio.Context()
    publisher = context.socket(zmq.XPUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    event_endpoint = f"tcp://127.0.0.1:{port}"
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    try:
        endpoint = runtime.endpoint(f"mooncake-{uuid.uuid4().hex}.worker.generate")
        await endpoint.register_endpoint_instance()
        config = ModelRuntimeConfig()
        config.set_engine_specific(
            "sglang_hicache_mooncake",
            json.dumps(
                {
                    "backend": "mooncake",
                    "tenant_id": "default",
                    "page_size": 4,
                    "tp_size": 1,
                    "pp_size": 1,
                    "is_mla_model": False,
                    "is_eagle": False,
                    "should_split_heads": False,
                    "kv_events_endpoint": event_endpoint,
                }
            ),
        )
        await register_model(
            ModelInput.Tensor,
            ModelType.TensorBased,
            endpoint,
            "test-mooncake",
            runtime_config=config,
            worker_type=WorkerType.Aggregated,
            tensor_model_config={"name": "test-mooncake", "inputs": [], "outputs": []},
        )
        router = KvRouter(
            endpoint,
            4,
            KvRouterConfig(
                use_kv_events=True,
                shared_cache_type="hicache",
                enable_session_prefix_index=True,
            ),
        )
        control = router.mooncake_control(MooncakeClient(admin_url))
        async with asyncio.timeout(10):
            while await publisher.recv() != b"\x01":
                pass
        keys = [f"{42:016x}{'0' * 48}_0_{kind}" for kind in ("k", "v")]
        many_keys = [f"{43:016x}{'0' * 48}_{rank:05}_k" for rank in range(8193)]
        events = [
            {
                "event_type": "stored",
                "object_key": key,
                "tenant_id": "default",
                "medium": "cpu",
                "session_id": "s" if key in keys else "large",
            }
            for key in keys + many_keys
        ]
        await publisher.send_multipart(
            [b"", (1).to_bytes(8, "big"), msgpack.packb([0, events, 0])]
        )
        async with asyncio.timeout(10):
            while control.get_session_blocks("large") != [[43]]:
                await asyncio.sleep(0.01)
        report = await control.pin_session("s", ttl_ms=60000)
        assert report == {
            "key_results": [
                {"key": keys[0], "error_code": 0},
                {"key": keys[1], "error_code": -704},
            ],
            "unresolved_hashes": [],
        }
        assert requests[-1] == (
            "/api/v1/soft_pin",
            {"tenant_id": "default", "keys": keys, "ttl_ms": 60000},
        )
        assert await control.unpin_blocks([42, 99]) == report["key_results"]
        assert requests[-1] == (
            "/api/v1/soft_unpin",
            {"tenant_id": "default", "keys": keys},
        )
        count = len(requests)
        with pytest.raises(Exception, match="no observed Mooncake keys"):
            await control.pin_blocks([42, 99], ttl_ms=1)
        assert len(requests) == count
        failures.add(count + 2)
        with pytest.raises(MooncakeCommandError) as caught:
            await control.pin_session("large", ttl_ms=1000)
        error = caught.value
        assert len(error.completed_results) == 4096
        assert error.uncertain_keys == many_keys[4096:8192]
        assert error.unattempted_keys == many_keys[8192:]
        assert error.unresolved_hashes == []
        assert len(requests) == count + 2
    finally:
        runtime.shutdown()
        publisher.close()
        context.term()

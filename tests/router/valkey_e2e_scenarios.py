# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable cross-frontend Valkey routing scenarios."""

import asyncio
import contextlib
import json
import time

import aiohttp

from tests.router.common import valkey_index_key
from tests.router.helper import (
    get_runtime,
    poll_for_worker_instances,
    wait_for_frontend_ready,
)
from tests.router.router_process import FrontendRouterProcess


def _test_valkey_three_frontend_routing(
    engine_workers,
    block_size: int,
    request,
    frontend_ports: list[int],
    test_payload: dict,
    valkey_urls: str,
    valkey_index_scope: str,
    router_valkey_config: str,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Exercise one shared Valkey index through three independent frontends.

    Each of four mock workers receives a distinct forced prefix through the
    first frontend. The other two frontends must subsequently select that same
    worker for an annotation-only lookup, proving that they read the shared
    persistent index rather than a process-local radix tree.
    """
    assert len(frontend_ports) == 3

    def valkey_stats(url: str, key: str) -> tuple[int, int, int]:
        """Read DYNKV.STATS through the test's module-loaded Valkey endpoint."""
        import socket

        endpoint = url.removeprefix("valkey://").removeprefix("redis://")
        host, port = endpoint.rstrip("/").rsplit(":", 1)
        parts = (b"DYNKV.STATS", key.encode())
        request = bytearray(f"*{len(parts)}\r\n".encode())
        for part in parts:
            request.extend(f"${len(part)}\r\n".encode())
            request.extend(part)
            request.extend(b"\r\n")

        def read_line(sock: socket.socket) -> bytes:
            line = bytearray()
            while not line.endswith(b"\r\n"):
                chunk = sock.recv(1)
                if not chunk:
                    raise RuntimeError("unexpected EOF reading Valkey reply")
                line.extend(chunk)
            return bytes(line[:-2])

        with socket.create_connection((host, int(port)), timeout=2) as sock:
            sock.sendall(request)
            if sock.recv(1) != b"*":
                raise RuntimeError("DYNKV.STATS did not return an array")
            count = int(read_line(sock))
            values = []
            for _ in range(count):
                if sock.recv(1) != b":":
                    raise RuntimeError("DYNKV.STATS did not return integer entries")
                values.append(int(read_line(sock)))
        if len(values) != 3:
            raise RuntimeError(f"unexpected DYNKV.STATS response: {values}")
        return tuple(values)  # type: ignore[return-value]

    async def request_worker(
        port: int,
        payload: dict,
        *,
        forced_worker_id: int | None = None,
        query_only: bool = False,
    ) -> int:
        headers: dict[str, str] = {}
        if forced_worker_id is not None:
            headers["x-dynamo-worker-instance-id"] = str(forced_worker_id)
            headers["x-dynamo-dp-rank"] = "0"

        request_payload = {
            **payload,
            "stream": True,
            "nvext": (
                {"annotations": ["query_instance_id:"]}
                if query_only
                else {"extra_fields": ["worker_id"]}
            ),
        }
        url = f"http://localhost:{port}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=request_payload, headers=headers
            ) as response:
                body = await response.text()
                assert response.status == 200, (
                    f"frontend {port} returned {response.status}: {body}"
                )

        for part in body.split("\n\n"):
            data = part.strip()
            if not data.startswith("data:"):
                continue
            encoded = data.removeprefix("data:").strip()
            if encoded == "[DONE]":
                continue
            try:
                message = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            worker_info = message.get("nvext", {}).get("worker_id", {})
            worker_id = worker_info.get("prefill_worker_id")
            if worker_id is not None:
                return int(worker_id)
        raise AssertionError(f"frontend {port} did not return nvext.worker_id: {body}")

    async def wait_for_shared_route(
        port: int, payload: dict, expected_worker_id: int
    ) -> None:
        last_worker_id: int | None = None
        for _ in range(60):
            last_worker_id = await request_worker(port, payload, query_only=True)
            if last_worker_id == expected_worker_id:
                return
            await asyncio.sleep(0.25)
        raise AssertionError(
            f"frontend {port} never selected cache owner {expected_worker_id}; "
            f"last selection was {last_worker_id}"
        )

    async def wait_for_block_growth(
        primary_url: str, index_key: str, previous_blocks: int
    ) -> int:
        observed = previous_blocks
        for _ in range(100):
            observed = await asyncio.to_thread(valkey_stats, primary_url, index_key)
            if observed[0] > previous_blocks:
                return observed[0]
            await asyncio.sleep(0.05)
        raise AssertionError(
            "forced worker request did not publish a new Valkey KV block; "
            f"previous={previous_blocks}, observed={observed}"
        )

    with contextlib.ExitStack() as stack:
        for port in frontend_ports:
            stack.enter_context(
                FrontendRouterProcess(
                    request,
                    block_size,
                    port,
                    engine_workers.namespace,
                    store_backend,
                    request_plane=request_plane,
                    min_initial_workers=engine_workers.num_workers,
                    router_valkey_config=router_valkey_config,
                    router_replica_sync=True,
                    event_plane="zmq",
                )
            )

        index_key = valkey_index_key(
            engine_workers.namespace,
            engine_workers.component_name,
            valkey_index_scope,
            block_size,
        )
        primary_url, replica_url = valkey_urls.split(",", maxsplit=1)

        async def run() -> None:
            for port in frontend_ports:
                await wait_for_frontend_ready(
                    frontend_url=f"http://localhost:{port}",
                    expected_num_workers=engine_workers.num_workers,
                    timeout=120,
                    engine_workers=engine_workers,
                    store_backend=store_backend,
                    request_plane=request_plane,
                )

            runtime = get_runtime(
                store_backend=store_backend, request_plane=request_plane
            )
            endpoint = runtime.endpoint(
                f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
            )
            worker_ids = sorted(
                await poll_for_worker_instances(
                    endpoint, engine_workers.num_workers, max_wait_time=120
                )
            )
            assert len(worker_ids) == engine_workers.num_workers

            block_count = 0
            for index, worker_id in enumerate(worker_ids):
                # The repo-local hermetic tokenizer intentionally has no text
                # vocabulary. These model-agnostic marker strings are ordinary
                # text for production tokenizers and distinct registered tokens
                # for that fixture, so both paths create full, unique KV blocks.
                marker = f"<|reserved_special_token_{index}|> "
                payload = {
                    **test_payload,
                    "messages": [
                        {
                            **test_payload["messages"][0],
                            "content": (
                                f"{test_payload['messages'][0]['content']} "
                                f"valkey-shared-prefix-worker-{index} "
                                + marker
                                * (block_size * 4)
                            ),
                        }
                    ],
                }
                selected = await request_worker(
                    frontend_ports[0], payload, forced_worker_id=worker_id
                )
                assert selected == worker_id, (
                    f"forced request selected {selected}, expected {worker_id}"
                )
                block_count = await wait_for_block_growth(
                    primary_url, index_key, block_count
                )
                for port in frontend_ports[1:]:
                    await wait_for_shared_route(port, payload, worker_id)

        asyncio.run(run())

        # The primary and replica hold the same native module index before the
        # frontend contexts exit. This also proves the worker-side direct
        # writer topology is safe to acknowledge through WAIT.
        primary_stats = valkey_stats(primary_url, index_key)
        assert primary_stats[0] > 0
        assert primary_stats[1] >= engine_workers.num_workers
        for _ in range(40):
            replica_stats = valkey_stats(replica_url, index_key)
            if replica_stats[:2] == primary_stats[:2]:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                f"Valkey replica did not converge: primary={primary_stats}, replica={replica_stats}"
            )

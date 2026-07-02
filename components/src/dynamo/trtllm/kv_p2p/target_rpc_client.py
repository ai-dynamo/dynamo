# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-side RPC client living in the dynamo worker parent process.

When the target worker's connector wants to resolve a remote-G2 plan,
it calls this client which discovers the chosen source worker via
Dynamo and then uses direct ZMQ/TCP for the side-effecting remote-G2
control RPCs.

This module only contains the **parent-side async client wrapper**.
How that wrapper is reached from the engine-subprocess connector (which
is in a different process and can't share Python objects with the
parent) is a Stage 3 concern — that's where the engine→parent IPC
bridge lives.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional


class RemoteG2DirectControlClient:
    """Direct TCP control client used by the staged remote-G2 RPC path.

    The target sends resolve/metadata/release requests to the direct
    endpoint discovered through Dynamo.
    """

    def __init__(self, timeout_ms: int = 5000) -> None:
        self._timeout_ms = timeout_ms

    def request(
        self,
        address: str,
        method: str,
        payload: dict,
        expected_identity: Optional[dict] = None,
    ) -> Optional[dict]:
        import pickle

        import zmq

        ctx = zmq.Context.instance()
        req = ctx.socket(zmq.REQ)
        req.RCVTIMEO = self._timeout_ms
        req.SNDTIMEO = self._timeout_ms
        try:
            req.connect(self._normalize_address(address))
            request = {"method": method, "payload": payload}
            if expected_identity is not None:
                request["expected_identity"] = expected_identity
            req.send(pickle.dumps(request))
            raw = req.recv()
            response = pickle.loads(raw)
            return response if isinstance(response, dict) else None
        except Exception:
            logging.exception(
                "remote_g2: direct control request failed method=%s address=%s",
                method,
                address,
            )
            return None
        finally:
            req.close(linger=0)

    @staticmethod
    def _normalize_address(address: str) -> str:
        if address.startswith("tcp://"):
            return address
        return f"tcp://{address}"


class _TargetRpcClient:
    """Wraps the dynamo runtime client for making remote-G2 RPC calls.

    Holds a lazily-initialized Dynamo client for discovery. Hot
    remote-G2 RPCs use the discovered direct ZMQ/TCP endpoint so they do
    not depend on Dynamo's streamed response path.
    """

    def __init__(self, runtime: Any, namespace: str, component: str) -> None:
        self._runtime = runtime
        self._namespace = namespace
        self._component = component
        # Lazy-init: the Dynamo client is only needed for direct endpoint discovery.
        self._control_info_client: Optional[Any] = None
        self._direct_control_client = RemoteG2DirectControlClient()
        self._direct_control_info_by_source: dict[int, dict] = {}
        self._client_init_lock = asyncio.Lock()

    async def _prime_discovery(self, client: Any, endpoint_name: str) -> None:
        """Wait until dynamo discovery surfaces at least one instance for
        the freshly-built client. ``client.instance_ids()`` is a sync
        cache read; immediately after ``ep.client()`` returns, the cache
        is typically empty even though etcd already has the instances
        registered. Without this prime, the first ``direct()`` call
        raises ``instance_id=X not found for endpoint ...``."""
        try:
            await asyncio.wait_for(client.wait_for_instances(), timeout=10.0)
        except asyncio.TimeoutError:
            logging.warning(
                "remote_g2: %s discovery prime timed out — no instances visible",
                endpoint_name,
            )

    async def _control_info_endpoint_client(self) -> Any:
        if self._control_info_client is None:
            async with self._client_init_lock:
                if self._control_info_client is None:
                    ep = self._runtime.endpoint(
                        f"{self._namespace}.{self._component}.remote-g2-control-info"
                    )
                    client = await ep.client()
                    await self._prime_discovery(client, "remote-g2-control-info")
                    self._control_info_client = client
        return self._control_info_client

    async def get_direct_control_info(
        self, source_worker_id: int
    ) -> Optional[dict]:
        """Fetch the source worker's direct-control address.

        Dynamo remains the bootstrap/discovery layer. The returned direct
        endpoint is cached before side-effecting remote-G2 requests are
        sent to it.
        """
        cached = self._direct_control_info_by_source.get(source_worker_id)
        if cached is not None:
            return cached

        try:
            client = await self._control_info_endpoint_client()
        except Exception:
            logging.exception(
                "remote_g2: failed to obtain control-info endpoint client"
            )
            return None

        try:
            stream = await client.direct({}, instance_id=source_worker_id)
            async for response in stream:
                if hasattr(response, "data") and callable(response.data):
                    response = response.data()
                if not isinstance(response, dict) or not response.get("ok"):
                    return None
                result = response.get("result")
                if not isinstance(result, dict):
                    return None
                address = result.get("address")
                if not isinstance(address, str) or not address:
                    return None
                control_info = dict(result)
                self._direct_control_info_by_source[source_worker_id] = control_info
                return control_info
        except Exception:
            logging.exception(
                "remote_g2: control-info RPC to source_worker_id=%s failed",
                source_worker_id,
            )
            return None
        return None

    async def _direct_request(
        self, source_worker_id: int, method: str, payload: dict
    ) -> Optional[dict]:
        control_info = await self.get_direct_control_info(source_worker_id)
        if control_info is None:
            return None
        address = control_info.get("address")
        if not isinstance(address, str) or not address:
            return None

        expected_identity = control_info.get("identity")
        if isinstance(expected_identity, dict):
            expected_identity = dict(expected_identity)
        else:
            expected_identity = {}
        expected_identity["worker_id"] = source_worker_id

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            self._direct_control_client.request,
            address,
            method,
            payload,
            expected_identity,
        )
        if response is None:
            self._direct_control_info_by_source.pop(source_worker_id, None)
        return response

    async def get_source_metadata(
        self,
        source_worker_id: int,
        peer_name: str = "",
        peer_connection_info: str = "",
    ) -> Optional[dict]:
        """Fetch the source worker's NIXL agent identity for transfer setup.

        When ``peer_name`` + ``peer_connection_info`` are provided, the
        source side will pre-load us as a NIXL peer before returning its
        metadata. The side-effecting RPC is sent over the direct ZMQ/TCP
        control endpoint after Dynamo-based endpoint discovery.
        """
        payload: dict = {}
        if peer_name:
            payload["peer_name"] = peer_name
        if peer_connection_info:
            payload["peer_connection_info"] = peer_connection_info

        response = await self._direct_request(
            source_worker_id, "get_metadata", payload
        )
        if not isinstance(response, dict):
            return None
        if not response.get("ok"):
            logging.warning(
                "remote_g2: get_source_metadata returned not-ok: %s",
                response.get("error"),
            )
            return None
        inner = response.get("result")
        return inner if isinstance(inner, dict) else None

    async def resolve_and_lease(
        self, plan: dict, source_worker_id: int
    ) -> Optional[dict]:
        """Resolve and lease source blocks over the direct control path.

        Returns the source response dict. ``None`` means transport or
        discovery failed; callers should reject the plan and fall back to
        local recompute.
        """
        return await self._direct_request(
            source_worker_id, "resolve_and_lease", {"plan": plan}
        )

    async def release_lease(
        self, lease_id: str, source_worker_id: int, reason: str = "ack"
    ) -> Optional[dict]:
        """Release a source lease over the direct control path."""
        return await self._direct_request(
            source_worker_id,
            "release_lease",
            {"lease_id": lease_id, "reason": reason},
        )


def build_target_rpc_client(
    runtime: Any, namespace: str, component: str
) -> _TargetRpcClient:
    """Construct the target-side RPC client. Cheap — no network I/O at
    construction; the per-endpoint dynamo clients are lazy-initialized
    on first call.
    """
    return _TargetRpcClient(runtime, namespace, component)

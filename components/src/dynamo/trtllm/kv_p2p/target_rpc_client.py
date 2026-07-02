# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-side RPC client living in the dynamo worker parent process.

When the target worker's connector wants to resolve a remote-G2 plan,
it calls this client which routes to the chosen source worker via
``client.direct(payload, instance_id=source_worker_id)`` against the
source's ``remote-g2-resolve`` endpoint.

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

    Stage 1 uses this only for ``identify``. Side-effecting operations
    remain on the existing path until endpoint discovery and identity
    validation are proven independently.
    """

    def __init__(self, timeout_ms: int = 5000) -> None:
        self._timeout_ms = timeout_ms

    def identify(
        self, address: str, expected_worker_id: int | str | None = None
    ) -> Optional[dict]:
        response = self.request(address, "identify", {})
        if not isinstance(response, dict) or not response.get("ok"):
            return None

        result = response.get("result")
        if not isinstance(result, dict):
            # Backward-compatible shape from the original direct-ZMQ spike.
            result = response

        worker_id = result.get("worker_id")
        if expected_worker_id is not None and str(worker_id) != str(
            expected_worker_id
        ):
            logging.warning(
                "remote_g2: direct identify worker mismatch expected=%s actual=%s address=%s",
                expected_worker_id,
                worker_id,
                address,
            )
            return None
        return result

    def request(self, address: str, method: str, payload: dict) -> Optional[dict]:
        import pickle

        import zmq

        ctx = zmq.Context.instance()
        req = ctx.socket(zmq.REQ)
        req.RCVTIMEO = self._timeout_ms
        req.SNDTIMEO = self._timeout_ms
        try:
            req.connect(self._normalize_address(address))
            req.send(pickle.dumps({"method": method, "payload": payload}))
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

    Holds a lazily-initialized dynamo client per endpoint. Calls are
    async and route via ``client.direct(payload, instance_id=...)`` so
    they reach the specific source worker the plan names.
    """

    def __init__(self, runtime: Any, namespace: str, component: str) -> None:
        self._runtime = runtime
        self._namespace = namespace
        self._component = component
        # Lazy-init: clients are constructed on first use of each endpoint.
        self._resolve_client: Optional[Any] = None
        self._release_client: Optional[Any] = None
        self._metadata_client: Optional[Any] = None
        self._control_info_client: Optional[Any] = None
        self._direct_control_client = RemoteG2DirectControlClient()
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

    async def _resolve_endpoint_client(self) -> Any:
        if self._resolve_client is None:
            async with self._client_init_lock:
                if self._resolve_client is None:
                    ep = self._runtime.endpoint(
                        f"{self._namespace}.{self._component}.remote-g2-resolve"
                    )
                    client = await ep.client()
                    await self._prime_discovery(client, "remote-g2-resolve")
                    self._resolve_client = client
        return self._resolve_client

    async def _release_endpoint_client(self) -> Any:
        if self._release_client is None:
            async with self._client_init_lock:
                if self._release_client is None:
                    ep = self._runtime.endpoint(
                        f"{self._namespace}.{self._component}.remote-g2-release"
                    )
                    client = await ep.client()
                    await self._prime_discovery(client, "remote-g2-release")
                    self._release_client = client
        return self._release_client

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

    async def _metadata_endpoint_client(self) -> Any:
        if self._metadata_client is None:
            async with self._client_init_lock:
                if self._metadata_client is None:
                    ep = self._runtime.endpoint(
                        f"{self._namespace}.{self._component}.remote-g2-metadata"
                    )
                    client = await ep.client()
                    await self._prime_discovery(client, "remote-g2-metadata")
                    self._metadata_client = client
        return self._metadata_client

    async def get_direct_control_info(
        self, source_worker_id: int
    ) -> Optional[dict]:
        """Fetch and validate the source worker's direct-control address.

        This remains non-side-effecting: the Dynamo runtime endpoint only
        returns the advertised direct endpoint, and this method then calls
        direct ``identify`` to prove the endpoint belongs to the requested
        source worker.
        """
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
                identity = self._direct_control_client.identify(
                    address, expected_worker_id=source_worker_id
                )
                if identity is None:
                    return None
                return {"address": address, "identity": identity}
        except Exception:
            logging.exception(
                "remote_g2: control-info RPC to source_worker_id=%s failed",
                source_worker_id,
            )
            return None
        return None

    async def get_source_metadata(
        self,
        source_worker_id: int,
        peer_name: str = "",
        peer_connection_info: str = "",
    ) -> Optional[dict]:
        """Fetch the source worker's NIXL agent identity for transfer setup.

        When ``peer_name`` + ``peer_connection_info`` are provided, the
        source side will pre-load us as a NIXL peer (bidirectional
        handshake) before returning its metadata. UCX needs both peers
        to know each other's connection info for the subsequent
        createXferReq's rkey lookup to work.

        Returns the inner result dict (with ``remote_name``, ``agent_desc``,
        ``source_generation``, ``source_worker_id``, ``source_dp_rank``) on
        success, or ``None`` on transport/RPC failure.
        """
        try:
            client = await self._metadata_endpoint_client()
        except Exception:
            logging.exception(
                "remote_g2: failed to obtain metadata endpoint client"
            )
            return None

        payload: dict = {}
        if peer_name:
            payload["peer_name"] = peer_name
        if peer_connection_info:
            payload["peer_connection_info"] = peer_connection_info
        try:
            stream = await client.direct(payload, instance_id=source_worker_id)
            async for response in stream:
                if hasattr(response, "data") and callable(response.data):
                    response = response.data()
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
        except Exception:
            logging.exception(
                "remote_g2: get_source_metadata RPC to source_worker_id=%s failed",
                source_worker_id,
            )
            return None
        return None

    async def resolve_and_lease(
        self, plan: dict, source_worker_id: int
    ) -> Optional[dict]:
        """Call the source worker's resolve endpoint. Returns the response
        dict (whose ``ok`` field indicates success). Returns ``None`` on
        transport-level failure (caller should treat as plan rejected and
        fall back to local recompute).
        """
        try:
            client = await self._resolve_endpoint_client()
        except Exception:
            logging.exception(
                "remote_g2: failed to obtain resolve endpoint client"
            )
            return None

        payload = {"plan": plan}
        try:
            # PROBE: enumerate which instance_ids dynamo discovery sees
            # for this endpoint so we can compare against the plan's
            # source_worker_id.
            try:
                ids = list(client.instance_ids())
            except Exception:
                ids = ["<query_failed>"]
            logging.warning(
                "PROBE rpc_chain target_client_direct instance_id=%s endpoint=remote-g2-resolve known_ids=%s",
                source_worker_id,
                ids,
            )
            # client.direct(...) returns a Future that resolves to the
            # response stream — await it to get the async iterator.
            stream = await client.direct(payload, instance_id=source_worker_id)
            async for response in stream:
                # Unary RPC — the source endpoint yields exactly once.
                # Dynamo wraps yielded values in an `Annotated` object
                # (Rust-side container, not picklable). Unwrap to the
                # plain dict the source endpoint actually yielded.
                # Unwrap dynamo's Annotated wrapper. `.data` is a method
                # on the Rust-side object, not an attribute — call it to
                # get the plain Python dict the source endpoint yielded.
                if hasattr(response, "data") and callable(response.data):
                    response = response.data()
                return response
        except Exception:
            logging.exception(
                "remote_g2: resolve_and_lease RPC to source_worker_id=%s failed",
                source_worker_id,
            )
            return None
        # Empty stream — endpoint closed without yielding anything.
        return None

    async def release_lease(
        self, lease_id: str, source_worker_id: int, reason: str = "ack"
    ) -> Optional[dict]:
        """Call the source worker's release endpoint. Same semantics as
        ``resolve_and_lease`` — ``None`` on transport failure."""
        try:
            client = await self._release_endpoint_client()
        except Exception:
            logging.exception(
                "remote_g2: failed to obtain release endpoint client"
            )
            return None

        payload = {"lease_id": lease_id, "reason": reason}
        try:
            stream = await client.direct(payload, instance_id=source_worker_id)
            async for response in stream:
                # Unwrap dynamo's Annotated wrapper. `.data` is a method
                # on the Rust-side object, not an attribute — call it to
                # get the plain Python dict the source endpoint yielded.
                if hasattr(response, "data") and callable(response.data):
                    response = response.data()
                return response
        except Exception:
            logging.exception(
                "remote_g2: release_lease RPC to source_worker_id=%s failed",
                source_worker_id,
            )
            return None
        return None


def build_target_rpc_client(
    runtime: Any, namespace: str, component: str
) -> _TargetRpcClient:
    """Construct the target-side RPC client. Cheap — no network I/O at
    construction; the per-endpoint dynamo clients are lazy-initialized
    on first call.
    """
    return _TargetRpcClient(runtime, namespace, component)

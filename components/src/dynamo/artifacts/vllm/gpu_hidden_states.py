# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident hidden-state export for vLLM's `extract_hidden_states` method.

Experimental. `DynamoGpuHiddenStatesConnector` replaces the example connector's
device-to-host copy and safetensors write with a device-to-device gather into a
GPU tensor that stays in the worker. The scheduler returns only a handle in
`kv_transfer_params["hidden_states_handle"]`; the handle's owner then claims it
through `collective_rpc("dynamo_claim_hidden_states", ...)`, served by
`DynamoHiddenStateWorkerExtension`, and receives `dynamo.nixl_connect` read
metadata. A consumer pulls the rows straight from worker GPU memory.

Load it with::

    --kv-transfer-config '{"kv_connector": "DynamoGpuHiddenStatesConnector",
      "kv_connector_module_path": "dynamo.artifacts.vllm.gpu_hidden_states",
      "kv_role": "kv_producer"}'
    --worker-extension-cls \
      dynamo.artifacts.vllm.gpu_hidden_states.DynamoHiddenStateWorkerExtension

Reading over CUDA IPC without NVLink: UCX disables CUDA IPC GETs unless the GPUs
are NVLink-connected ("cuda-ipc get zcopy disabled as no nvlinks detected"), so a
NIXL read of GPU memory falls back to software emulation over loopback TCP. Set
`UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y` in the *reading* process (the worker needs no
change) before it creates its first NIXL agent. On a GB10, that took a 1 GiB
GPU-to-GPU read from 3.5 s to 1.2 s, almost all of it connection setup. Measure
on PCIe topologies before relying on it: UCX's default exists for those.

Each gathered tensor is freed after one read, when its lease ends, or when nobody
claims it within `claim_timeout_s`. `max_held_bytes` caps GPU memory held.
Extra config keys: `max_held_bytes` (default 2 GiB), `claim_timeout_s` (30),
`max_lease_s` (300). Single tensor-parallel rank only for now.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    PendingSave,
    extract_from_kv_cache,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

logger = init_logger(__name__)

HANDLE_KEY = "hidden_states_handle"
_DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


@dataclass
class _Gathered:
    tensor: torch.Tensor | None
    token_ids: list[int]
    event: torch.cuda.Event | None
    nbytes: int
    error: str | None = None
    claimed: bool = False
    operation: Any = field(default=None, repr=False)


class DynamoGpuHiddenStatesConnector(ExampleHiddenStatesConnector):
    """Keeps extracted hidden states on the GPU and serves them over NIXL."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        extra = self._kv_transfer_config
        self._max_held_bytes = int(
            extra.get_from_extra_config("max_held_bytes", 2 << 30)
        )
        self._claim_timeout_s = float(
            extra.get_from_extra_config("claim_timeout_s", 30.0)
        )
        self._max_lease_s = int(extra.get_from_extra_config("max_lease_s", 300))
        self._held: dict[str, _Gathered] = {}
        self._held_bytes = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._nixl_connector = None

    # ---- scheduler side -------------------------------------------------

    def request_finished(self, request, block_ids):
        delay, params = super().request_finished(request, block_ids)
        if params is None:
            return delay, params
        # The base class names a file this connector never writes.
        return delay, {HANDLE_KEY: request.request_id}

    # ---- worker side ----------------------------------------------------

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        if self._is_tp_rank_zero:
            # Start the NIXL agent (about a second) on the private loop now, so
            # the first claim, which runs on vLLM's engine thread, doesn't wait.
            self._call_soon(self._warm_nixl)

    async def _warm_nixl(self) -> None:
        try:
            await self._nixl_connection()
        except Exception:  # noqa: BLE001 -- the first claim retries and reports
            logger.exception("Starting the hidden-state NIXL agent failed")

    async def _nixl_connection(self):
        import dynamo.nixl_connect as nixl_connect

        if self._nixl_connector is None:
            self._nixl_connector = nixl_connect.Connector()
        # Idempotent; nixl_connect has no public accessor for its shared agent.
        return await self._nixl_connector._create_connection()

    def wait_for_save(self) -> None:
        # The example connector pre-creates lock files here; nothing to lock.
        return

    def _submit_async_write(self, pending: PendingSave) -> None:
        if not self._is_tp_rank_zero:
            return
        assert self._kv_cache is not None
        block_ids = torch.tensor(pending.block_ids, dtype=torch.long)
        offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            offsets.reshape((1, self._block_size))
            + block_ids.reshape((block_ids.shape[0], 1)) * self._block_size
        ).flatten()
        num_tokens = pending.token_ids.shape[0]
        row_bytes = (
            self._kv_cache.shape[1]
            * self._kv_cache.shape[3]
            * self._kv_cache.element_size()
        )
        nbytes = num_tokens * row_bytes
        token_ids = [int(token) for token in pending.token_ids.tolist()]
        with self._lock:
            over_budget = self._held_bytes + nbytes > self._max_held_bytes
            if not over_budget:
                self._held_bytes += nbytes
        if over_budget:
            logger.warning(
                "Hidden-state GPU budget exhausted; dropping %s (%d bytes)",
                pending.req_id,
                nbytes,
            )
            self._remember(
                pending.req_id,
                _Gathered(None, token_ids, None, 0, error="gpu budget exhausted"),
            )
            return
        copy_stream = self._get_copy_stream()
        ready = torch.cuda.Event()
        ready.record()
        copy_stream.wait_event(ready)
        with torch.cuda.stream(copy_stream):
            with gpu_sync_allowed():
                slots = slot_mapping.to(device=self._kv_cache.device, non_blocking=True)
            # Advanced indexing allocates a new tensor, so the rows outlive the
            # KV blocks, which the scheduler frees once this copy completes.
            gathered = extract_from_kv_cache(self._kv_cache, slots, num_tokens)
        done = torch.cuda.Event()
        done.record(copy_stream)
        # The base get_finished reports the request done sending (freeing its
        # blocks) once this event completes.
        self._req_copy_events[pending.req_id] = done
        self._remember(pending.req_id, _Gathered(gathered, token_ids, done, nbytes))

    def _remember(self, req_id: str, gathered: _Gathered) -> None:
        with self._lock:
            self._held[req_id] = gathered
        self._call_later(self._claim_timeout_s, self._expire_unclaimed, req_id)

    # ---- claims (called through the worker extension) --------------------

    def claim(self, req_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """Return NIXL read metadata once the gather is done. Never blocks long."""
        with self._lock:
            gathered = self._held.get(req_id)
            if gathered is None:
                # Not ours: under data parallelism another engine owns it.
                return None
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= self._max_lease_s
        ):
            return {"state": "failed", "error": "invalid lease"}
        with self._lock:
            gathered = self._held.get(req_id)
            if gathered is None:
                return None
            if gathered.error is not None:
                self._held.pop(req_id, None)
                return {"state": "failed", "error": gathered.error}
            if gathered.claimed:
                return {"state": "failed", "error": "already claimed"}
            if gathered.event is not None and not gathered.event.query():
                return {"state": "pending"}
            gathered.claimed = True
        tensor = gathered.tensor
        assert tensor is not None
        try:
            metadata = self._run(self._register(req_id, tensor))
        except Exception as exc:  # noqa: BLE001 -- reported to the claimer
            logger.exception("Registering hidden states %s with NIXL failed", req_id)
            self._release(req_id)
            return {"state": "failed", "error": f"nixl registration failed: {exc}"}
        self._call_soon(self._hold, req_id, lease_seconds)
        expires = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        return {
            "state": "ready",
            "rdma_metadata": metadata,
            "dtype": _DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype)),
            "shape": list(tensor.shape),
            "byte_count": gathered.nbytes,
            "token_ids": gathered.token_ids,
            "expires_at": expires.isoformat().replace("+00:00", "Z"),
        }

    async def _register(self, req_id: str, tensor: torch.Tensor):
        import dynamo.nixl_connect as nixl_connect

        await self._nixl_connection()
        operation = await self._nixl_connector.create_readable(
            nixl_connect.Descriptor(tensor)
        )
        with self._lock:
            gathered = self._held.get(req_id)
            if gathered is not None:
                gathered.operation = operation
        return operation.metadata().model_dump(mode="json")

    async def _hold(self, req_id: str, lease_seconds: int) -> None:
        with self._lock:
            gathered = self._held.get(req_id)
        if gathered is None or gathered.operation is None:
            return
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                gathered.operation.wait_for_completion(), lease_seconds
            )
            logger.debug(
                "Hidden states %s read after %.3fs", req_id, time.monotonic() - started
            )
        except asyncio.TimeoutError:
            logger.warning("Hidden-state lease for %s expired unread", req_id)
        finally:
            self._release(req_id)

    def _expire_unclaimed(self, req_id: str) -> None:
        with self._lock:
            gathered = self._held.get(req_id)
            if gathered is None or gathered.claimed:
                return
        logger.warning("Hidden states %s were never claimed; freeing", req_id)
        self._release(req_id)

    def _release(self, req_id: str) -> None:
        with self._lock:
            gathered = self._held.pop(req_id, None)
            if gathered is None:
                return
            self._held_bytes -= gathered.nbytes
        if gathered.operation is not None:
            gathered.operation.__exit__(None, None, None)
        gathered.tensor = None

    # ---- private event loop for NIXL and timers -------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            ready = threading.Event()
            holder: dict[str, asyncio.AbstractEventLoop] = {}

            def main():
                loop = asyncio.new_event_loop()
                holder["loop"] = loop
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            threading.Thread(
                target=main, name="dynamo-hidden-state-nixl", daemon=True
            ).start()
            ready.wait(5)
            self._loop = holder["loop"]
            return self._loop

    def _run(self, coroutine, timeout: float = 30.0):
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout)

    def _call_soon(self, coroutine_fn, *args) -> None:
        loop = self._ensure_loop()
        asyncio.run_coroutine_threadsafe(coroutine_fn(*args), loop)

    def _call_later(self, delay: float, fn, *args) -> None:
        loop = self._ensure_loop()
        loop.call_soon_threadsafe(loop.call_later, delay, fn, *args)


def _find_connector() -> DynamoGpuHiddenStatesConnector | None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return None
    group = get_kv_transfer_group()
    for candidate in (group, *getattr(group, "_connectors", ())):
        if isinstance(candidate, DynamoGpuHiddenStatesConnector):
            return candidate
    return None


class DynamoHiddenStateWorkerExtension:
    """Worker methods reached through `collective_rpc` by name."""

    def dynamo_claim_hidden_states(
        self, handle: str, lease_seconds: int
    ) -> dict[str, Any] | None:
        connector = _find_connector()
        return None if connector is None else connector.claim(handle, lease_seconds)


def all_engines_rpc(async_llm):
    """A `collective_rpc` that returns every engine's worker results.

    With data parallelism, vLLM's `AsyncLLM.collective_rpc` runs on every engine
    but returns only the first engine's results, so a handle owned by another
    engine would be claimed there and the answer discarded. The engine client's
    `call_utility_all_async` returns all of them; without data parallelism this
    falls back to `collective_rpc`.
    """
    call_all = getattr(async_llm.engine_core, "call_utility_all_async", None)
    if call_all is None:
        return async_llm.collective_rpc

    async def rpc(method, timeout=None, args=(), kwargs=None):
        per_engine = await call_all("collective_rpc", method, timeout, args, kwargs)
        return [result for engine_results in per_engine for result in engine_results]

    return rpc


async def claim_hidden_states(
    rpc, handle: str, lease_seconds: int, *, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    """Poll the workers until the handle is ready; `rpc` is `collective_rpc`.

    From a Dynamo handler, pass `all_engines_rpc(engine_client)`: plain
    `AsyncLLM.collective_rpc` misses handles owned by any engine but the first
    under data parallelism. Returns the single worker result that knows the handle.
    """
    deadline = time.monotonic() + timeout_seconds
    delay = 0.002
    while True:
        results = await rpc("dynamo_claim_hidden_states", args=(handle, lease_seconds))
        known = [result for result in results if result is not None]
        if len(known) > 1:
            raise RuntimeError("hidden-state handle claimed on several workers")
        if known and known[0]["state"] != "pending":
            return known[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("hidden states were not ready before the timeout")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.05)

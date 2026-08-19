# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot-aware vLLM NIXL connector loaded via ``kv_connector_module_path``.

CRIU cannot restore IB/UCX QPs. After restore, live prefill still caches the
old ``engine_id``. This connector shuts down the restored NIXL agent and side
channel, mints ``engine_id`` from the restore ``incarnation_id``, re-registers
KV tensors, and republishes handshake metadata so prefill treats decode as new.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

from dynamo.common.snapshot.constants import (
    SNAPSHOT_CONTROL_DIR,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.restore_context import (
    RestoreIdentity,
    load_restore_identity,
    should_rebind_pd,
)

logger = logging.getLogger(__name__)

SNAPSHOT_NIXL_MODULE = "dynamo.vllm.snapshot_nixl"
NIXL_CONNECTOR_NAMES = frozenset(
    {"NixlConnector", "NixlPullConnector", "NixlPushConnector"}
)
MULTI_CONNECTOR_NAMES = frozenset({"MultiConnector", "PdConnector"})
_HANDSHAKE_DIR = "nixl-handshake"
_HANDSHAKE_WAIT_SEC = 30.0
_HANDSHAKE_POLL_SEC = 0.05


def configure_snapshot_nixl_connector(kv_cfg: Any) -> bool:
    """Point NixlConnector entries at this module. Keep the public class name."""

    if kv_cfg is None:
        return False
    rewritten = False
    connector_name = getattr(kv_cfg, "kv_connector", None)
    if connector_name in NIXL_CONNECTOR_NAMES:
        kv_cfg.kv_connector_module_path = SNAPSHOT_NIXL_MODULE
        rewritten = True
    elif connector_name in MULTI_CONNECTOR_NAMES:
        extra = getattr(kv_cfg, "kv_connector_extra_config", None) or {}
        for entry in extra.get("connectors", []):
            if (
                isinstance(entry, dict)
                and entry.get("kv_connector") in NIXL_CONNECTOR_NAMES
            ):
                entry["kv_connector_module_path"] = SNAPSHOT_NIXL_MODULE
                rewritten = True
    if rewritten:
        logger.info(
            "Snapshot mode: loading NIXL connector from %s", SNAPSHOT_NIXL_MODULE
        )
    return rewritten


def engine_id_for_restore(old_engine_id: str | None, incarnation_id: str) -> str:
    """Keep a trailing ``_dpN`` suffix so data-parallel ranks stay distinct."""

    if old_engine_id and "_dp" in old_engine_id:
        suffix = old_engine_id[old_engine_id.rfind("_dp") :]
        return f"{incarnation_id}{suffix}"
    return incarnation_id


def handshake_path(
    incarnation_id: str,
    pp_rank: int,
    tp_rank: int,
    control_dir: str | None = None,
) -> Path:
    root = control_dir or os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    return Path(root) / _HANDSHAKE_DIR / incarnation_id / f"{pp_rank}-{tp_rank}.msgpack"


def write_handshake_metadata(
    incarnation_id: str,
    pp_rank: int,
    tp_rank: int,
    metadata: object,
    control_dir: str | None = None,
) -> Path:
    path = handshake_path(incarnation_id, pp_rank, tp_rank, control_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(pickle.dumps(metadata))
    tmp_path.replace(path)
    return path


def read_handshake_metadata(
    incarnation_id: str,
    pp_size: int,
    tp_size: int,
    control_dir: str | None = None,
    timeout: float = _HANDSHAKE_WAIT_SEC,
) -> dict[tuple[int, int], object]:
    expected = {(pp, tp) for pp in range(pp_size) for tp in range(tp_size)}
    deadline = time.monotonic() + timeout
    while True:
        found: dict[tuple[int, int], object] = {}
        for key in expected:
            path = handshake_path(incarnation_id, key[0], key[1], control_dir)
            if path.is_file():
                found[key] = pickle.loads(path.read_bytes())
        if found.keys() == expected:
            return found
        if time.monotonic() >= deadline:
            missing = expected - found.keys()
            raise RuntimeError(
                "timed out waiting for NIXL handshake metadata after snapshot "
                f"restore; missing ranks {sorted(missing)}"
            )
        time.sleep(_HANDSHAKE_POLL_SEC)


def apply_restore_side_channel_host(identity: RestoreIdentity) -> None:
    host = identity.side_channel_host
    if host:
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host


class SnapshotNixlMixin:
    """Rebuild the NIXL worker/scheduler after CRIU restore."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_incarnation_id: str | None = None
        self._registered_kv_caches: dict[str, Any] | None = None
        self._registered_cross_layer_kv: tuple[Any, Any] | None = None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._registered_kv_caches = kv_caches
        super().register_kv_caches(kv_caches)  # type: ignore[misc]

    def register_cross_layers_kv_cache(self, kv_cache: Any, attn_backend: Any) -> None:
        self._registered_cross_layer_kv = (kv_cache, attn_backend)
        super().register_cross_layers_kv_cache(kv_cache, attn_backend)  # type: ignore[misc]

    def get_handshake_metadata(self) -> Any:
        self._maybe_rebind()
        return super().get_handshake_metadata()  # type: ignore[misc]

    def reset_cache(self) -> bool | None:
        self._maybe_rebind()
        result = super().reset_cache()  # type: ignore[misc]
        return True if result is None else result

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        self._maybe_rebind()
        return super().start_load_kv(*args, **kwargs)  # type: ignore[misc]

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        self._maybe_rebind()
        return super().build_connector_meta(scheduler_output)  # type: ignore[misc]

    def _maybe_rebind(self) -> None:
        identity = load_restore_identity()
        if not should_rebind_pd(self._bound_incarnation_id, identity):
            return
        assert identity is not None
        logger.info(
            "Rebinding vLLM NIXL after snapshot restore incarnation_id=%s",
            identity.incarnation_id,
        )
        apply_restore_side_channel_host(identity)
        new_engine_id = engine_id_for_restore(self.engine_id, identity.incarnation_id)
        self._vllm_config.kv_transfer_config.engine_id = new_engine_id
        if self.connector_worker is not None:
            self._rebind_worker(identity, new_engine_id)
        if self.connector_scheduler is not None:
            self._rebind_scheduler(identity, new_engine_id)
        self.engine_id = new_engine_id
        self._bound_incarnation_id = identity.incarnation_id

    def _rebind_worker(self, identity: RestoreIdentity, new_engine_id: str) -> None:
        worker_cls = type(self.connector_worker)
        self.shutdown()
        self.connector_worker = worker_cls(
            self._vllm_config, new_engine_id, self.kv_cache_config
        )
        if self._registered_kv_caches is not None:
            self.connector_worker.register_kv_caches(self._registered_kv_caches)
        if self._registered_cross_layer_kv is not None:
            kv_cache, _attn_backend = self._registered_cross_layer_kv
            self.connector_worker.register_cross_layers_kv_caches(kv_cache)
        metadata = self.connector_worker.xfer_handshake_metadata
        pp_rank, tp_rank = _local_rank_key(self)
        write_handshake_metadata(identity.incarnation_id, pp_rank, tp_rank, metadata)
        logger.info(
            "Wrote NIXL handshake metadata incarnation_id=%s pp=%s tp=%s engine_id=%s",
            identity.incarnation_id,
            pp_rank,
            tp_rank,
            new_engine_id,
        )

    def _rebind_scheduler(self, identity: RestoreIdentity, new_engine_id: str) -> None:
        scheduler_cls = type(self.connector_scheduler)
        self.shutdown()
        self.connector_scheduler = scheduler_cls(
            self._vllm_config, new_engine_id, self.kv_cache_config
        )
        parallel = self._vllm_config.parallel_config
        metadata = read_handshake_metadata(
            identity.incarnation_id,
            getattr(parallel, "pipeline_parallel_size", 1) or 1,
            getattr(parallel, "tensor_parallel_size", 1) or 1,
        )
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)
        logger.info(
            "Scheduler NIXL listener rebound incarnation_id=%s engine_id=%s host=%s",
            identity.incarnation_id,
            new_engine_id,
            identity.side_channel_host,
        )


def _local_rank_key(connector: Any) -> tuple[int, int]:
    worker = getattr(connector, "connector_worker", None)
    tp_rank = getattr(worker, "tp_rank", None)
    if tp_rank is None:
        tp_rank = getattr(connector._vllm_config.parallel_config, "rank", 0) or 0
    try:
        from vllm.distributed.parallel_state import get_pp_group

        pp_rank = get_pp_group().rank_in_group
    except Exception:
        pp_rank = (
            getattr(connector._vllm_config.parallel_config, "pipeline_parallel_rank", 0)
            or 0
        )
    return int(pp_rank), int(tp_rank)


def __getattr__(name: str) -> Any:
    # vLLM's factory does getattr(module, kv_connector). Keep the stock names
    # and import the bases only when the engine actually constructs them.
    if name not in {
        "NixlConnector",
        "NixlPullConnector",
        "NixlPushConnector",
        "SnapshotNixlPullConnector",
        "SnapshotNixlPushConnector",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector as _NixlPullConnector,
        NixlPushConnector as _NixlPushConnector,
    )

    class SnapshotNixlPullConnector(SnapshotNixlMixin, _NixlPullConnector):
        pass

    class SnapshotNixlPushConnector(SnapshotNixlMixin, _NixlPushConnector):
        pass

    exported = {
        "SnapshotNixlPullConnector": SnapshotNixlPullConnector,
        "SnapshotNixlPushConnector": SnapshotNixlPushConnector,
        "NixlPullConnector": SnapshotNixlPullConnector,
        "NixlPushConnector": SnapshotNixlPushConnector,
        "NixlConnector": SnapshotNixlPullConnector,
    }
    globals().update(exported)
    return exported[name]


async def rebind_vllm_nixl_after_restore(engine: Any) -> None:
    """Eager worker-then-scheduler rebind after CRIU resume and wake."""

    if not should_rebind_pd(None):
        return
    identity = load_restore_identity()
    assert identity is not None
    logger.info(
        "Triggering vLLM NIXL rebind after snapshot restore incarnation_id=%s",
        identity.incarnation_id,
    )
    await engine.collective_rpc("get_kv_connector_handshake_metadata")
    await engine.reset_prefix_cache(reset_connector=True)

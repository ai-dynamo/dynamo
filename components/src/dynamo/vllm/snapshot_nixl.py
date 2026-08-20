# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot NIXL connector for vLLM, loaded via ``kv_connector_module_path``.

After restore, workers rebuild the NIXL agent and write handshake metadata under
the snapshot-control dir. The scheduler then shuts down the old side channel
and serves those new blobs. Prefill keys peers by ``engine_id``, so the
incarnation becomes the new id.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

from dynamo.common.snapshot.constants import (
    SNAPSHOT_CONTROL_DIR,
    SNAPSHOT_CONTROL_DIR_ENV,
)
from dynamo.common.snapshot.restore_context import (
    apply_snapshot_restore_env,
    load_restore_incarnation_id,
    should_rebind_pd,
)

logger = logging.getLogger(__name__)

SNAPSHOT_NIXL_MODULE = "dynamo.vllm.snapshot_nixl"
_NIXL_NAMES = frozenset({"NixlConnector", "NixlPullConnector", "NixlPushConnector"})
_WRAPPER_NAMES = frozenset({"MultiConnector", "PdConnector"})


def configure_snapshot_nixl_connector(kv_cfg: Any) -> None:
    """Point NixlConnector entries at this module. Keep the public class name."""

    if kv_cfg is None:
        return
    if kv_cfg.kv_connector in _NIXL_NAMES:
        kv_cfg.kv_connector_module_path = SNAPSHOT_NIXL_MODULE
    elif kv_cfg.kv_connector in _WRAPPER_NAMES:
        for entry in (kv_cfg.kv_connector_extra_config or {}).get("connectors", []):
            if isinstance(entry, dict) and entry.get("kv_connector") in _NIXL_NAMES:
                entry["kv_connector_module_path"] = SNAPSHOT_NIXL_MODULE


def _handshake_dir(incarnation_id: str, control_dir: str | None = None) -> Path:
    root = control_dir or os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    return Path(root) / "nixl-handshake" / incarnation_id


def write_handshake_metadata(
    incarnation_id: str,
    pp_rank: int,
    tp_rank: int,
    metadata: object,
    control_dir: str | None = None,
) -> None:
    directory = _handshake_dir(incarnation_id, control_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pp_rank}-{tp_rank}"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(pickle.dumps(metadata))
    tmp.replace(path)


def read_handshake_metadata(
    incarnation_id: str, control_dir: str | None = None
) -> dict[tuple[int, int], object]:
    directory = _handshake_dir(incarnation_id, control_dir)
    found = {
        tuple(int(part) for part in path.name.split("-")): pickle.loads(
            path.read_bytes()
        )
        for path in directory.glob("*-*")
        if path.is_file()
    }
    if not found:
        raise RuntimeError(
            f"no NIXL handshake metadata for incarnation {incarnation_id}"
        )
    return found  # type: ignore[return-value]


class SnapshotNixlMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_incarnation_id: str | None = None
        self._kv_caches: dict[str, Any] | None = None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._kv_caches = kv_caches
        super().register_kv_caches(kv_caches)  # type: ignore[misc]

    def get_handshake_metadata(self) -> Any:
        self._maybe_rebind()
        return super().get_handshake_metadata()  # type: ignore[misc]

    def reset_cache(self) -> bool | None:
        self._maybe_rebind()
        result = super().reset_cache()  # type: ignore[misc]
        return True if result is None else result

    def _maybe_rebind(self) -> None:
        if not should_rebind_pd(self._bound_incarnation_id):
            return
        incarnation_id = load_restore_incarnation_id()
        assert incarnation_id is not None
        apply_snapshot_restore_env()
        dp = getattr(self._vllm_config.parallel_config, "data_parallel_index", 0) or 0
        engine_id = f"{incarnation_id}_dp{dp}"
        self._vllm_config.kv_transfer_config.engine_id = engine_id
        logger.info(
            "Rebinding vLLM NIXL incarnation_id=%s engine_id=%s",
            incarnation_id,
            engine_id,
        )
        if self.connector_worker is not None:
            worker_cls = type(self.connector_worker)
            self.shutdown()
            self.connector_worker = worker_cls(
                self._vllm_config, engine_id, self.kv_cache_config
            )
            if self._kv_caches is not None:
                self.connector_worker.register_kv_caches(self._kv_caches)
            tp = getattr(self.connector_worker, "tp_rank", 0) or 0
            write_handshake_metadata(
                incarnation_id,
                0,
                int(tp),
                self.connector_worker.xfer_handshake_metadata,
            )
        if self.connector_scheduler is not None:
            scheduler_cls = type(self.connector_scheduler)
            self.shutdown()
            self.connector_scheduler = scheduler_cls(
                self._vllm_config, engine_id, self.kv_cache_config
            )
            self.connector_scheduler.set_xfer_handshake_metadata(
                read_handshake_metadata(incarnation_id)
            )
        self.engine_id = engine_id
        self._bound_incarnation_id = incarnation_id


def __getattr__(name: str) -> Any:
    if name not in {"NixlConnector", "NixlPullConnector", "NixlPushConnector"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPullConnector as _Pull,
        NixlPushConnector as _Push,
    )

    class NixlPullConnector(SnapshotNixlMixin, _Pull):
        pass

    class NixlPushConnector(SnapshotNixlMixin, _Push):
        pass

    exported = {
        "NixlPullConnector": NixlPullConnector,
        "NixlPushConnector": NixlPushConnector,
        "NixlConnector": NixlPullConnector,
    }
    globals().update(exported)
    return exported[name]


async def rebind_vllm_nixl_after_restore(engine: Any) -> None:
    if not should_rebind_pd(None):
        return
    await engine.collective_rpc("get_kv_connector_handshake_metadata")
    await engine.reset_prefix_cache(reset_connector=True)

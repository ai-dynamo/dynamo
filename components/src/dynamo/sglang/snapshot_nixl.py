# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot NIXL manager for SGLang.

SGLang has no connector module-path. Snapshot mode wraps ``get_kv_class`` so
capture constructs this manager. Spawned children re-import sglang; the wheel
``.pth`` calls :func:`install_snapshot_nixl` when ``DYN_SNAPSHOT_CONTROL_DIR``
is set. After restore a new agent name is required because prefill ignores a
repeat ``agent_name``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from dynamo.common.snapshot.restore_context import should_rebind_pd
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SnapshotNixlKVManager(NixlKVManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_incarnation_id: str | None = None
        self._nixl_agent = self.__dict__.pop("agent")

    def __getattr__(self, name: str) -> Any:
        if name == "agent":
            self._maybe_rebind()
            return self._nixl_agent
        raise AttributeError(name)

    def _add_remote_peer(self, decode_kv_args: Any) -> None:
        self._maybe_rebind()
        super()._add_remote_peer(decode_kv_args)

    def _maybe_rebind(self) -> None:
        if not should_rebind_pd(self._bound_incarnation_id):
            return
        from dynamo.common.snapshot.restore_context import load_restore_incarnation_id
        from nixl._api import nixl_agent, nixl_agent_config

        incarnation_id = load_restore_incarnation_id()
        assert incarnation_id is not None
        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        self._nixl_agent = nixl_agent(
            str(uuid.uuid4()),
            nixl_agent_config(
                backends=[backend],
                num_threads=(
                    8 if self.disaggregation_mode == DisaggregationMode.PREFILL else 0
                ),
            ),
        )
        self._bound_incarnation_id = incarnation_id
        self.register_buffer_to_engine()
        logger.info(
            "Rebinding SGLang NIXL incarnation_id=%s agent_name=%s",
            incarnation_id,
            self._nixl_agent.name,
        )


def install_snapshot_nixl() -> None:
    from sglang.srt.disaggregation import utils as disagg_utils
    from sglang.srt.disaggregation.utils import KVClassType, TransferBackend

    original = disagg_utils.get_kv_class
    if getattr(original, "_dyn_snapshot_nixl", False):
        return

    def get_kv_class(transfer_backend: Any, class_type: Any) -> Any:
        cls = original(transfer_backend, class_type)
        if (
            transfer_backend == TransferBackend.NIXL
            and class_type == KVClassType.MANAGER
        ):
            return SnapshotNixlKVManager
        return cls

    get_kv_class._dyn_snapshot_nixl = True  # type: ignore[attr-defined]
    disagg_utils.get_kv_class = get_kv_class

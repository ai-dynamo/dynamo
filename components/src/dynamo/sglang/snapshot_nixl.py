# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot-aware SGLang NIXL manager.

SGLang has no connector module-path factory. Snapshot mode wraps
``get_kv_class`` so capture constructs this manager, and spawned children pick
up the wrap via ``dynamo_snapshot.pth``. After restore, a new NIXL agent name
is required: live prefill ignores a repeat ``agent_name``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from dynamo.common.snapshot.restore_context import (
    load_restore_identity,
    should_rebind_pd,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.utils.network import get_zmq_socket_on_host

logger = logging.getLogger(__name__)


class SnapshotNixlKVManager(NixlKVManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_incarnation_id: str | None = None

    def _add_remote_peer(self, decode_kv_args: Any) -> None:
        self._maybe_rebind()
        super()._add_remote_peer(decode_kv_args)

    def _maybe_rebind(self) -> None:
        identity = load_restore_identity()
        if not should_rebind_pd(self._bound_incarnation_id, identity):
            return
        assert identity is not None

        from nixl._api import nixl_agent, nixl_agent_config

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        agent_config = nixl_agent_config(
            backends=[backend],
            num_threads=(
                8 if self.disaggregation_mode == DisaggregationMode.PREFILL else 0
            ),
        )
        new_name = str(uuid.uuid4())
        logger.info(
            "Rebinding SGLang NIXL after snapshot restore incarnation_id=%s "
            "agent_name=%s",
            identity.incarnation_id,
            new_name,
        )
        self.agent = nixl_agent(new_name, agent_config)
        if identity.pod_ip and identity.pod_ip != self.local_ip:
            self._rebind_zmq(identity.pod_ip)
        elif identity.pod_ip:
            self.local_ip = identity.pod_ip
        self.register_buffer_to_engine()
        self._bound_incarnation_id = identity.incarnation_id

    def _rebind_zmq(self, new_ip: str) -> None:
        import zmq

        try:
            self.server_socket.close(linger=0)
        except Exception:
            logger.exception("Failed to close restored SGLang NIXL ZMQ socket")
        self.local_ip = new_ip
        context = zmq.Context()
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=self.local_ip
        )
        logger.info(
            "Rebound SGLang NIXL ZMQ listener to %s:%s",
            self.local_ip,
            self.rank_port,
        )


class SnapshotNixlKVReceiver(NixlKVReceiver):
    def send_metadata(self, *args: Any, **kwargs: Any) -> None:
        self.kv_mgr._maybe_rebind()
        return super().send_metadata(*args, **kwargs)

    def _register_kv_args(self) -> None:
        self.kv_mgr._maybe_rebind()
        return super()._register_kv_args()


def install_snapshot_nixl() -> None:
    """Register snapshot NIXL classes with SGLang's ``get_kv_class``."""

    from sglang.srt.disaggregation import utils as disagg_utils
    from sglang.srt.disaggregation.utils import KVClassType, TransferBackend

    original = disagg_utils.get_kv_class
    if getattr(original, "_dyn_snapshot_nixl", False):
        return

    def get_kv_class(transfer_backend: Any, class_type: Any) -> Any:
        cls = original(transfer_backend, class_type)
        if transfer_backend != TransferBackend.NIXL:
            return cls
        if class_type == KVClassType.MANAGER:
            return SnapshotNixlKVManager
        if class_type == KVClassType.RECEIVER:
            return SnapshotNixlKVReceiver
        return cls

    get_kv_class._dyn_snapshot_nixl = True  # type: ignore[attr-defined]
    disagg_utils.get_kv_class = get_kv_class
    logger.info("Installed snapshot SGLang NIXL manager/receiver")

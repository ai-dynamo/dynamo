# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL connector that holds completed KV receives for a fixed time.

A decode worker loads it with ``kv_connector="NixlConnector"`` and
``kv_connector_module_path`` pointing at this module. The real NIXL read runs,
but its completion is reported to the scheduler only after
``dynamo_test_recv_hold_s`` from ``kv_connector_extra_config``, so the request
stays in ``WAITING_FOR_REMOTE_KVS`` with its blocks allocated.

The class must keep the name ``NixlConnector``: vLLM
(``has_connector("NixlConnector")``) and Dynamo (``_uses_nixl_connector``)
match the literal name to set up NIXL.
"""

import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.logger import init_logger

# Under the vllm logger hierarchy so the lines reach the worker log.
logger = init_logger(f"vllm.{__name__}")

HOLD_KEY = "dynamo_test_recv_hold_s"
DEFAULT_HOLD_S = 10.0


class NixlConnector(NixlPullConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hold_s = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                HOLD_KEY, DEFAULT_HOLD_S
            )
        )
        # request ID -> (first seen, receive failed)
        self._held: dict[str, tuple[float, bool]] = {}

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        results = super().get_transfer_results(finished_req_ids)
        now = time.monotonic()
        for req_id in results.finished_recving:
            logger.info("Holding KV receive for request %s", req_id)
            self._held[req_id] = (now, req_id in results.failed_recving)
        results.finished_recving = set()
        results.failed_recving = set()

        for req_id, (held_at, failed) in list(self._held.items()):
            if now - held_at < self._hold_s:
                continue
            del self._held[req_id]
            logger.info("Released held KV receive for request %s", req_id)
            results.finished_recving.add(req_id)
            if failed:
                results.failed_recving.add(req_id)
        return results

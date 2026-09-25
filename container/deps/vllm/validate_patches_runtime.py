# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the DeepSeek V4.1 Flash vLLM runtime patch stack.

The installed pinned-nightly wheel must contain the #58215 DeepSelect sentinel
bound, #57662's NIXL region geometry key, and #58038's telemetry-completion
behavior. This deliberately inspects the installed wheel, not a source tree.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.model_executor.kernels.attention.dsa import sparse_mqa_logits


def validate_dsa_sentinel_bound() -> None:
    source = inspect.getsource(sparse_mqa_logits)
    assert "valid = (c >= 0) & (c < width)" in source
    assert "        width," in source


def validate_nixl_region_key() -> None:
    source = inspect.getsource(base_worker)
    assert "region_key = (base_addr, block_len)" in source
    assert "if region_key in seen_region_keys:" in source
    assert "seen_region_keys.append(region_key)" in source


def main() -> int:
    validate_dsa_sentinel_bound()
    validate_nixl_region_key()

    worker = object.__new__(NixlBaseConnectorWorker)
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.side_effect = RuntimeError(
        "nixlNoTelemetryError"
    )
    worker.xfer_stats = MagicMock()
    worker._log_failure = MagicMock()
    worker._handle_failed_transfer = MagicMock(return_value=True)
    transfers = {"request": [101]}

    done_req_ids, failed_req_ids = worker._pop_done_transfers(transfers)

    assert done_req_ids == {"request"}
    assert failed_req_ids == set()
    assert not transfers
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(101)
    worker.xfer_stats.record_transfer.assert_not_called()
    worker._log_failure.assert_not_called()
    worker._handle_failed_transfer.assert_not_called()
    print("DSv4.1 Flash vLLM runtime patch validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

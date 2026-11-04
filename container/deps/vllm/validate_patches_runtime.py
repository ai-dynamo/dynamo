# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the DeepSeek V4.1 Flash vLLM runtime patch stack.

The pinned nightly supplies #58215's DeepSelect sentinel bound. The installed
wheel must also contain the #57662 NIXL region geometry key, #58038 telemetry
completion behavior, and #55374 piecewise-prefix load protocol. This deliberately
inspects the installed wheel, not a source tree.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker, metadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.model_executor.kernels.attention.dsa import sparse_mqa_logits


def validate_dsa_sentinel_bound() -> None:
    source = inspect.getsource(sparse_mqa_logits)
    assert "valid = (c >= 0) & (c < width)" in source
    assert "        width," in source


def validate_nixl_region_key() -> None:
    source = inspect.getsource(base_worker)
    assert "region_key = (base_addr, block_len)" in source
    assert "if region_key in seen_region_keys and not route_packed_layers:" in source
    assert "seen_region_keys.append(region_key)" in source


def validate_piecewise_prefix_loading() -> None:
    source = inspect.getsource(multi_connector)
    assert "load_policy=range_aware" in source
    assert "update_state_after_alloc_for_range" in source
    assert "self._request_load_ranges" in source
    assert metadata.NIXL_CONNECTOR_VERSION == 14

    read_blocks = inspect.signature(NixlPullConnectorWorker._read_blocks).parameters
    assert read_blocks["load_start_token"].default == 0
    assert read_blocks["load_end_token"].default == 0

    write_blocks = inspect.signature(NixlPushConnectorWorker._xfer_blocks).parameters
    assert write_blocks["load_start_token"].default == 0
    assert write_blocks["load_end_token"].default == 0

    worker = object.__new__(NixlBaseConnectorWorker)
    worker.block_size = 16
    assert worker._kernel_block_window(64, 224) == (4, 10)
    decode, prefill = worker._slice_attention_to_load_window(
        list(range(20)), list(range(30)), 64, 224
    )
    assert decode == list(range(10))
    assert prefill == list(range(4, 14))


def main() -> int:
    validate_dsa_sentinel_bound()
    validate_nixl_region_key()
    validate_piecewise_prefix_loading()

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

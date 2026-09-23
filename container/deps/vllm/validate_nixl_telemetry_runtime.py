# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate vLLM #58038's NIXL telemetry completion behavior.

The pinned nightly reports a transfer as ``DONE`` before vLLM reads its optional
telemetry.  A missing telemetry record must not turn that completed transfer
into a failed request.  This exercises the installed wheel, not a source tree.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)


def main() -> int:
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
    print("NIXL telemetry completion validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

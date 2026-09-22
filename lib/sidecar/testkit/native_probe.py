# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only observation of completed NIXL transfers in the native worker."""

import json
import os

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlKVConnectorStats

_record_transfer = NixlKVConnectorStats.record_transfer


def record_transfer(self, result):
    _record_transfer(self, result)
    payload = json.dumps({"bytes": result.totalBytes, "descriptors": result.descCount})
    with open(
        os.environ["SIDECAR_NATIVE_TRANSFER_PROBE"], "a", encoding="utf-8"
    ) as probe:
        probe.write(payload + "\n")


NixlKVConnectorStats.record_transfer = record_transfer


class TransferProbe:
    """Importing the extension installs the observer before model initialization."""

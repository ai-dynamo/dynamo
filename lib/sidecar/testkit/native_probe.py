# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only observations for native transfer, encoder-cache and NVTX checks."""

import json
import os
import time
from pathlib import Path

from nixl._api import nixl_agent

_transfer = nixl_agent.transfer


def transfer(self, *args, **kwargs):
    result = _transfer(self, *args, **kwargs)
    gate = Path(os.environ["SIDECAR_NATIVE_TRANSFER_GATE"])
    if (gate / "armed").exists():
        if result not in ("PROC", "DONE"):
            raise RuntimeError(f"native NIXL transfer was rejected: {result}")
        try:
            with (gate / "accepted").open("x", encoding="utf-8") as accepted:
                accepted.write(json.dumps({"state": result}))
            print(f"Native NIXL transfer accepted: {result}", flush=True)
        except FileExistsError:
            return result
        deadline = time.monotonic() + 30
        while not (gate / "release").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("native transfer gate was not released")
            time.sleep(0.005)
    return result


if os.environ.get("SIDECAR_NATIVE_TRANSFER_GATE"):
    nixl_agent.transfer = transfer

if os.environ.get("SIDECAR_NATIVE_BACKEND") == "vllm":
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
        NixlKVConnectorStats,
    )

    _record_transfer = NixlKVConnectorStats.record_transfer

    def record_transfer(self, result):
        _record_transfer(self, result)
        payload = json.dumps(
            {"bytes": result.totalBytes, "descriptors": result.descCount}
        )
        with open(
            os.environ["SIDECAR_NATIVE_TRANSFER_PROBE"], "a", encoding="utf-8"
        ) as probe:
            probe.write(payload + "\n")

    NixlKVConnectorStats.record_transfer = record_transfer

    if ec_probe := os.environ.get("SIDECAR_NATIVE_EC_PROBE"):
        from vllm.distributed.ec_transfer.ec_connector.example_connector import (
            ECExampleConnector,
        )

        _load_caches = ECExampleConnector.start_load_caches

        def start_load_caches(self, encoder_cache, **kwargs):
            previous = set(encoder_cache)
            result = _load_caches(self, encoder_cache, **kwargs)
            with open(ec_probe, "a", encoding="utf-8") as probe:
                for identifier in encoder_cache.keys() - previous:
                    tensor = encoder_cache[identifier]
                    probe.write(
                        json.dumps(
                            {
                                "identifier": identifier,
                                "numel": tensor.numel(),
                                "bytes": tensor.numel() * tensor.element_size(),
                            }
                        )
                        + "\n"
                    )
            return result

        ECExampleConnector.start_load_caches = start_load_caches

    if nvtx_probe := os.environ.get("SIDECAR_NATIVE_NVTX_PROBE"):
        import torch.cuda.nvtx as nvtx

        _range_push = nvtx.range_push

        def range_push(message):
            result = _range_push(message)
            if "Module" in message:
                with open(nvtx_probe, "a", encoding="utf-8") as probe:
                    probe.write(json.dumps(message) + "\n")
            return result

        nvtx.range_push = range_push


class TransferProbe:
    """Importing the extension installs the observer before model initialization."""

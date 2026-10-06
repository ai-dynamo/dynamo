# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only observation of native executor cleanup and KV-transfer completion."""

import asyncio
import contextvars
import json
import os
import sys
import threading
import time
from contextlib import aclosing
from datetime import datetime
from pathlib import Path

import yaml
from tensorrt_llm.commands.serve import main
from tensorrt_llm.grpc.openengine import servicer

_sampling_params = servicer.sampling_params_from_request
_format_result = servicer._format_result
_init = servicer.OpenEngineInferenceServicer.__init__
_generate = servicer.OpenEngineInferenceServicer.Generate
_request_id = contextvars.ContextVar("request_id", default=None)
_lock = threading.Lock()


def record(event):
    with _lock:
        with open(
            os.environ["DYN_TEST_TRANSFER_PROBE"], "a", encoding="utf-8"
        ) as output:
            output.write(json.dumps({"port": _port, **event}) + "\n")


def init_with_stats(self, llm, *args, **kwargs):
    _init(self, llm, *args, **kwargs)
    generate = llm.generate_async

    def submit(*args, **kwargs):
        record(
            {
                "kind": "submitted",
                "request_id": _request_id.get(),
                "submitted_at": time.time(),
            }
        )
        return generate(*args, **kwargs)

    llm.generate_async = submit

    def collect():
        try:
            while True:
                for stats in llm.get_stats(timeout=0.1):
                    record(
                        {
                            "kind": "stats",
                            "batch_started_at": datetime.strptime(
                                stats["timestamp"], "%m-%d-%Y %H:%M:%S.%f"
                            ).timestamp(),
                            "stats": {
                                "iter": stats["iter"],
                                "numQueuedRequests": stats["numQueuedRequests"],
                                "requestStats": [
                                    {
                                        key: req[key]
                                        for key in ("id", "stage", "numGeneratedTokens")
                                    }
                                    for req in stats.get("requestStats", [])
                                ],
                                "kvCacheStats": {
                                    "usedNumBlocks": stats["kvCacheStats"][
                                        "usedNumBlocks"
                                    ]
                                },
                            },
                        }
                    )
        except Exception as error:
            record({"kind": "stats_error", "error": repr(error)})

    threading.Thread(target=collect, daemon=True).start()


async def generate_with_gate(self, request, context):
    token = _request_id.set(request.request_id)
    try:
        gated = False
        gate_request = Path(os.environ["DYN_TEST_TRANSFER_PROBE"] + ".request")
        async with aclosing(_generate(self, request, context)) as responses:
            async for response in responses:
                # Hold the first decode token after transfer, before the sidecar
                # receives it. Keep the native generator alive during cancellation.
                if (
                    not gated
                    and gate_request.exists()
                    and request.request_id == gate_request.read_text()
                    and request.kv.HasField("session")
                    and response.WhichOneof("event") == "token"
                ):
                    gated = True
                    gate = Path(os.environ["DYN_TEST_TRANSFER_PROBE"] + ".release")
                    record({"kind": "decode_gate", "request_id": request.request_id})
                    async with asyncio.timeout(10):
                        while not gate.exists():
                            await asyncio.sleep(0.01)
                yield response
    finally:
        record({"kind": "stream_closed", "request_id": request.request_id})
        _request_id.reset(token)


def sampling_params_with_metrics(*args, **kwargs):
    params = _sampling_params(*args, **kwargs)
    params.return_perf_metrics = True
    return params


def record_transfer(result, state):
    responses = _format_result(result, state)
    if result.finished:
        for output in result.outputs:
            metrics = output.request_perf_metrics
            if metrics is None:
                continue
            timing = metrics.timing_metrics
            start = timing.kv_cache_transfer_start.total_seconds()
            end = timing.kv_cache_transfer_end.total_seconds()
            if end > start > 0 and timing.kv_cache_size > 0:
                event = {
                    "request_id": state.request_id,
                    "bytes": timing.kv_cache_size,
                    "transfer_seconds": end - start,
                }
                record({"kind": "transfer", **event})
    return responses


if __name__ == "__main__":
    _port = int(sys.argv[sys.argv.index("--port") + 1])
    # Extend the launcher's complete config, preserving its KV budget and NIXL.
    config_index = (
        max(i for i, arg in enumerate(sys.argv) if arg == "--extra_llm_api_options") + 1
    )
    config = yaml.safe_load(Path(sys.argv[config_index]).read_text())
    config.update(enable_iter_perf_stats=True, enable_iter_req_stats=True)
    config.setdefault("kv_cache_config", {})["enable_block_reuse"] = False
    config_path = Path(os.environ["DYN_TEST_TRANSFER_PROBE"] + f".{_port}.yaml")
    config_path.write_text(yaml.safe_dump(config))
    sys.argv[config_index] = str(config_path)
    servicer.OpenEngineInferenceServicer.__init__ = init_with_stats
    servicer.OpenEngineInferenceServicer.Generate = generate_with_gate
    servicer.sampling_params_from_request = sampling_params_with_metrics
    servicer._format_result = record_transfer
    main()

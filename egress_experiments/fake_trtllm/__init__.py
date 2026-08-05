# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stub of TRT-LLM that is faithful at exactly one boundary: the one dynamo
touches.

The engine itself is opaque -- it is a child process that sleeps for an
iteration and emits tokens. Everything *around* it is modelled from the real
sources, because that is what the dynamo worker's asyncio loop actually pays
for:

===========================  =========================================================
this package                 real TRT-LLM
===========================  =========================================================
``aqueue.AsyncQueue``        ``tensorrt_llm/llmapi/utils.py:388``
``aqueue._SyncQueue``        ``tensorrt_llm/llmapi/utils.py:475`` (incl. ``notify_many``)
``result.GenerationResult``  ``tensorrt_llm/executor/result.py:949``
``result.._handle_response`` ``tensorrt_llm/executor/result.py:454``
``ipc.PairEndpoint``         ``tensorrt_llm/executor/ipc.py:497`` (``FusedIpcQueue``)
``engine.engine_main``       ``executor/base_worker.py:1117`` (``_AwaitResponseHelper``)
``engine._send_iteration``   ``executor/base_worker.py:1252`` (``handle_for_ipc_batched``)
``llm.FakeLLM.generate_async``   ``tensorrt_llm/llmapi/llm.py``
``llm._Proxy.dispatch_result_task``  ``tensorrt_llm/executor/proxy.py:532``
===========================  =========================================================
"""

from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import (
    CompletionOutput,
    GenerationResult,
    Response,
)

__all__ = [
    "AsyncQueue",
    "SyncQueue",
    "FakeLLM",
    "GenerationResult",
    "CompletionOutput",
    "Response",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-worker methods the launcher calls by name after a self-benchmark.

``AsyncLLM.collective_rpc`` reaches the model workers through the engine-core
client, which encodes the RPC with msgpack and refuses a Python callable
unless ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``; a method name always goes
through. vLLM adds the class named by ``--worker-extension-cls`` to every
model worker's bases, so a method defined here can be called as
``collective_rpc("<method name>")``.

Every model worker imports this module, so it must stay free of import-time
side effects.
"""

from __future__ import annotations

from typing import Any


class FpmBenchmarkWorkerExtension:
    """vLLM worker extension that undoes benchmark-only worker settings."""

    # Set by vLLM's WorkerBase. A bare annotation adds no class attribute, so it
    # cannot trip vLLM's check for names the worker class already has.
    vllm_config: Any

    def fpm_disable_cudagraph_metrics(self) -> dict[str, Any]:
        """Turn vLLM's ``cudagraph_metrics`` off in this worker.

        The model runner reads the option on every step, so the change takes
        effect from the next forward pass. The launcher calls this only when it
        found the option, so a worker without it fails the call.
        """
        self.vllm_config.observability_config.cudagraph_metrics = False
        return {"cudagraph_metrics": False}

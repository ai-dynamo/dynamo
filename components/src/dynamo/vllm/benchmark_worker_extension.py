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
    """vLLM worker extension for the launcher's post-benchmark worker calls."""

    # Set by vLLM's WorkerBase. A bare annotation adds no class attribute, so it
    # cannot trip vLLM's check for names the worker class already has.
    vllm_config: Any
    rank: int

    def fpm_disable_cudagraph_metrics(self) -> dict[str, Any]:
        """Turn vLLM's ``cudagraph_metrics`` off in this worker.

        The model runner reads the option on every step, so the change takes
        effect from the next forward pass. The launcher calls this only when it
        found the option, so a worker without ``vllm_config`` or its
        ``observability_config`` fails the call.
        """
        self.vllm_config.observability_config.cudagraph_metrics = False
        return {"cudagraph_metrics": False}

    def fpm_engine_probe(self) -> dict[str, Any]:
        """Report what this worker resolved: attention backends and graphs.

        vLLM chooses the attention backend inside the worker
        (``Attention.__init__`` -> ``get_attn_backend``) and never writes the
        result back into ``VllmConfig``, so the layer registry in
        ``compilation_config.static_forward_context`` is the only place the
        resolved names exist. Same attribute names in vLLM 0.28.0 to 0.30.0.
        A worker whose config lacks one of the fields read here fails the call,
        which the launcher records as a failed probe.
        """
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config
        backends: dict[str, str] = {}
        prefill_backends: list[str] = []
        capture_errors: dict[str, str] = {}
        for name, layer in compilation_config.static_forward_context.items():
            # Only attention layers define get_attn_backend; the registry also
            # holds other layers (MoE, for one).
            get_backend = getattr(layer, "get_attn_backend", None)
            if not callable(get_backend):
                continue
            try:
                backends[name] = get_backend().get_name()
            except Exception as error:
                # Not re-raised: the failed layer is kept in capture_errors and
                # the probe still reports the others.
                capture_errors[name] = f"{type(error).__name__}: {error}"
                continue
            # Only MLA layers have a prefill backend.
            prefill = getattr(layer, "prefill_backend", None)
            if prefill is not None:
                # MLA layers hold a backend instance; tolerate a bare class.
                prefill_class = prefill if isinstance(prefill, type) else type(prefill)
                prefill_backends.append(prefill_class.__name__)
        unique_prefill = sorted(set(prefill_backends))
        worker_rank = self.rank
        # vLLM zeroes parallel_config.data_parallel_rank on every engine for a
        # dense (non-MoE) model under external DP, keeping the true rank only
        # in data_parallel_index -- the same trap
        # InstrumentedScheduler._resolve_dp_rank already routes around, so
        # engine.resolved.dp_rank does not silently disagree with
        # engine.parallel.data_parallel_rank / dp.rank in the same artifact.
        dp_rank = parallel_config.data_parallel_index
        # No live process group is guaranteed where this method runs.
        # vLLM orders ranks as DP x PP x PCP x TP, so derive the model-parallel
        # ranks from the worker rank while keeping the explicit DP identity.
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        cudagraph_mode = compilation_config.cudagraph_mode
        return {
            "tp_rank": worker_rank % tp_size,
            "pp_rank": (worker_rank // (tp_size * pcp_size)) % pp_size,
            "pcp_rank": (worker_rank // tp_size) % pcp_size,
            "worker_rank": worker_rank,
            "dp_rank": dp_rank,
            "attention_backends": backends,
            "mla_prefill_backend": (
                unique_prefill[0] if len(unique_prefill) == 1 else None
            ),
            "cudagraph_mode_resolved": (
                None if cudagraph_mode is None else cudagraph_mode.name
            ),
            "cudagraph_capture_sizes_resolved": [
                int(size) for size in compilation_config.cudagraph_capture_sizes or []
            ],
            "capture_errors": capture_errors,
        }

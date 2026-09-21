# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-side initialization of private self-benchmark recurrent states.

Only the opt-in benchmark worker uses this class. Real attention prefixes are
untouched; random recurrent states are a synthetic performance input, not valid
model history. Preparation finishes in the discarded admission step.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

from dynamo.vllm.benchmark_points import RANDOM_KDA_BOUND, RANDOM_KDA_REQUEST_PREFIX


def fill_recurrent_states(
    states: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    *,
    request_id: str,
    layer_name: str,
    rank: int,
) -> None:
    """Fill only allocated state slots, without changing the model's RNG stream."""
    for tensor_index, state in enumerate(states):
        if not state.is_floating_point():
            raise ValueError(
                "Random benchmark state requires floating-point cache tensors"
            )
        seed_input = f"{request_id}:{layer_name}:{rank}:{tensor_index}".encode()
        seed = int.from_bytes(
            hashlib.blake2b(seed_input, digest_size=8).digest(), "little"
        )
        generator = torch.Generator(device=state.device).manual_seed(seed)
        for block_id in block_ids:
            if not 0 < block_id < state.shape[0]:
                raise ValueError(f"Invalid synthetic recurrent-state block: {block_id}")
            state[block_id].uniform_(
                -RANDOM_KDA_BOUND, RANDOM_KDA_BOUND, generator=generator
            )


class BenchmarkWorker(Worker):
    """Standard vLLM GPU worker with opt-in synthetic KDA admission."""

    _benchmark_kda_active = False

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_from_config(kv_cache_config)
        self._benchmark_kda_layers = []
        context = self.vllm_config.compilation_config.static_forward_context
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, MambaSpec):
                for name in group.layer_names:
                    self._benchmark_kda_layers.append(
                        (group_id, name, context[name].kv_cache)
                    )
        if not self._benchmark_kda_layers:
            raise ValueError(
                "--benchmark-randomize-kda-state requires recurrent cache groups"
            )
        # V1 wraps zeroing on the runner; V2 exposes the zeroer directly.
        # Native initialization can omit the V2 zeroer when the cache dtype
        # needs no automatic zeroing, but benchmark shadows still request it.
        zero_blocks = getattr(self.model_runner, "_zero_block_ids", None)
        if not callable(zero_blocks):
            zeroer = getattr(self.model_runner, "kv_block_zeroer", None)
            init_zeroer = getattr(self.model_runner, "_init_kv_zero_meta", None)
            if zeroer is None and callable(init_zeroer):
                init_zeroer()
                zeroer = getattr(self.model_runner, "kv_block_zeroer", None)
            zero_blocks = getattr(zeroer, "zero_block_ids", None)
        if not callable(zero_blocks):
            raise ValueError(
                "Random KDA benchmarking requires a vLLM GPU runner KV-zeroing API"
            )
        self._benchmark_zero_block_ids = zero_blocks
        self._benchmark_kda_active = True

    def finish_benchmark_kda_state(self) -> None:
        """Called before endpoint registration, including benchmark failure cleanup."""
        self._benchmark_kda_active = False
        self._benchmark_kda_layers = []

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if self._benchmark_kda_active:
            requests = [
                req
                for req in scheduler_output.scheduled_new_reqs
                if req.req_id.startswith(RANDOM_KDA_REQUEST_PREFIX)
            ]
            if requests:
                # Native zeroing normally runs inside execute_model. Do it now
                # and consume only this copy's zeroing list so it cannot erase
                # the random values before attention executes. In particular,
                # the typed KDA views alias the same backing pool as MLA.
                scheduler_output = copy.copy(scheduler_output)
                if scheduler_output.new_block_ids_to_zero:
                    self._benchmark_zero_block_ids(
                        scheduler_output.new_block_ids_to_zero
                    )
                    scheduler_output.new_block_ids_to_zero = None
                for req in requests:
                    for group_id, name, states in self._benchmark_kda_layers:
                        block_ids = sorted(set(req.block_ids[group_id]) - {0})
                        if not block_ids:
                            raise ValueError(
                                "Synthetic recurrent state must have a non-null block"
                            )
                        fill_recurrent_states(
                            states,
                            block_ids,
                            request_id=req.req_id,
                            layer_name=name,
                            rank=self.rank,
                        )
                # This work belongs to admission, never to the measured steady
                # steps, even when the runner returns asynchronous outputs.
                torch.cuda.current_stream(self.device).synchronize()
        return super().execute_model(scheduler_output)

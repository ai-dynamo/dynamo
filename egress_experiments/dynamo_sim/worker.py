# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dynamo TRT-LLM worker handler, reduced to the part the diagram measures.

This mirrors the response loop of
``components/src/dynamo/trtllm/request_handlers/handler_base.py``
(``_generate_locally_impl``), keeping the pieces that cost loop time and
dropping the ones that do not run on the decode path at all (multimodal,
disagg param codecs, logits processors, metrics).

What is kept verbatim, because it is the ``trtllm:build_response`` stage:

* the per-choice cursor ``output_tokens_per_choice``, since TRT-LLM streams
  CUMULATIVE ``token_ids`` per output and dynamo must emit only the new slice,
* ``finish_reason`` / ``stop_reason`` propagation and the "finished with no
  finish reason" fallback,
* ``completion_usage`` assembly on the final chunk,
* and the plain ``yield out``. That yield stays a yield even under push egress
  -- it is pure-Python generator delegation on one thread. The hop that push
  replaces is the OUTERMOST one, into Rust.

``generate`` is wrapped with the **real** ``push_egress_capable`` decorator
loaded from the repo (see :mod:`egress_experiments.dynamo_sim.realcode`), so
the pull/push fork under test is the shipped one, not a lookalike.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from egress_experiments.costs import Costs, pad_to, spin
from egress_experiments.dynamo_sim import realcode
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

_real = realcode.load_push_egress()

if _real is not None:
    push_egress_capable = _real.push_egress_capable
    USING_REAL_PUSH_EGRESS = True
else:  # pragma: no cover - only when the checkout is unavailable

    def push_egress_capable(func):  # type: ignore[misc]
        """Degenerate stand-in: pull path only."""

        def dispatch(self, request, context=None, response_sender=None, **kwargs):
            if response_sender is not None:
                raise RuntimeError("real push_egress.py unavailable; cannot push")
            return func(self, request, context, **kwargs)

        return dispatch

    USING_REAL_PUSH_EGRESS = False


@dataclass
class SamplingParams:
    """Only the fields ``FakeLLM.generate_async`` reads."""

    max_tokens: int = 64
    n: int = 1


class TrtllmWorkerHandler:
    """One handler instance serves every request, as in the real worker."""

    def __init__(
        self,
        llm: FakeLLM,
        costs: Optional[Costs] = None,
        records: Optional[Dict[int, RequestRecord]] = None,
    ) -> None:
        self.llm = llm
        self.costs = costs or Costs()
        self.records: Dict[int, RequestRecord] = records if records is not None else {}
        #: Responses this handler produced, for cross-checking egress counts.
        self.responses_yielded = 0

    # push_egress_capable must stay OUTERMOST -- the Rust opt-in check runs
    # inspect.signature() on the registered callable and needs to see
    # `response_sender`. Same constraint as AggregatedHandler.generate.
    @push_egress_capable
    async def generate(self, request: dict, context: Any) -> AsyncGenerator[dict, None]:
        async for out in self.generate_locally(request, context):
            yield out

    async def generate_locally(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        # Spans awaits, so start/end rather than push/pop -- same reason the
        # real handler uses range_decorator here.
        with range_("trtllm:generate_locally", color="blue"):
            async for out in self._generate_locally_impl(request, context):
                yield out

    async def _generate_locally_impl(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        generation_result, num_input_tokens = self._start_generation(request)
        output_tokens_per_choice: Dict[int, int] = {}

        # `async for` -> GenerationResult.__anext__ -> _aresult_step ->
        # aqueue.get() then _handle_response(). handle_response therefore runs
        # HERE, on the loop, inside this iteration.
        async for res in generation_result:
            for output in res.outputs:
                # trtllm:build_response -- inline, because npw=0. The range has
                # to span the dict work AND the padding, or a capture of this
                # run would report only the padding and under-read the stage.
                with range_("trtllm:build_response", color="yellow"):
                    start = _perf()

                    output_idx = getattr(output, "index", 0) or 0
                    tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                    next_total_toks = len(output.token_ids)

                    out: Dict[str, Any] = {
                        "token_ids": output.token_ids[tokens_so_far:],
                        "index": output_idx,
                    }
                    if output.finish_reason:
                        out["finish_reason"] = output.finish_reason
                    if output.stop_reason:
                        out["stop_reason"] = output.stop_reason

                    if out.get("finish_reason") or res.finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        total_completion_tokens = sum(
                            len(o.token_ids) for o in res.outputs
                        )
                        out["completion_usage"] = {
                            "prompt_tokens": int(num_input_tokens),
                            "completion_tokens": int(total_completion_tokens),
                            "total_tokens": int(
                                num_input_tokens + total_completion_tokens
                            ),
                            "prompt_tokens_details": None,
                        }

                    pad_to(start, self.costs.scaled(self.costs.build_response_us))

                self.responses_yielded += 1
                yield out
                output_tokens_per_choice[output_idx] = next_total_toks

    def _start_generation(self, request: dict, **response_path: Any):
        """Run the shared request path and submit before response handling forks."""
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            # The loop has finally drained to this request. Everything between
            # `accepted_ns` and here is time spent in the ONE asyncio deque --
            # the quantity the diagram leaves blank and queue_probe measures.
            record.admitted_ns = _perf()

        # The four pre-submit stages, under the same NVTX names the real
        # handler uses, so a capture of this run reads back identically.
        for stage_name, stage_us in (
            ("trtllm:normalize_request", self.costs.normalize_request_us),
            ("trtllm:setup_disagg_params", self.costs.setup_disagg_params_us),
            ("trtllm:prepare_input", self.costs.prepare_input_us),
            ("trtllm:sampling_params", self.costs.sampling_params_us),
        ):
            with range_(stage_name, color="cyan"):
                spin(self.costs.scaled(stage_us))

        sampling_params = SamplingParams(
            max_tokens=int(request.get("max_tokens", 64)),
            n=int(request.get("n", 1)),
        )

        # ---- the mocked boundary -----------------------------------------
        # The worker hands the request to the engine and gets a handle back.
        # Nothing about the engine is visible from here; responses arrive
        # asynchronously over the IPC lane.
        generation_result = self.llm.generate_async(
            inputs=request.get("token_ids"),
            sampling_params=sampling_params,
            disaggregated_params=None,
            streaming=True,
            trace_headers=None,
            scheduling_params=None,
            priority=0.5,
            cache_salt=None,
            **response_path,
        )

        num_input_tokens = len(request.get("token_ids") or [])
        return generation_result, num_input_tokens

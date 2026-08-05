# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage cost model, defaulted to the si=40 measurements.

Every default here is a **p50 measured on the capture itself** -- job 355778,
``decode_worker_0/nsys_355778_disagg_gen-rank0.sqlite``, decode rank 0,
``num_postprocess_workers: 0`` -- not a number read off the prose. Run
``python3 -m egress_experiments.capture_params <sqlite>`` to re-derive them.

That the diagram's figures are p50 and not mean is worth stating, because the
distributions are heavily skewed and it changes the headline::

    handle_response          mean 26.32   p50 23.97      <- diagram
    trtllm:build_response    mean 74.70   p50 50.66      <- diagram
    trtllm:push_send         mean 11.63   p50 10.72      <- diagram
    ------------------------------------------------
    3-stage total            mean 112.64  p50 85.35

The simulation does the real bookkeeping for each stage -- building the
response dict, slicing the token delta, walking the per-choice cursor -- and
then *pads* to the measured cost with :func:`spin`, a GIL-holding busy wait.

Padding rather than sleeping is the whole point. ``time.sleep`` releases the
GIL and yields the loop, which would erase exactly the effect under study: on
the real worker these stages are uninterruptible Python that the one event loop
must run to completion before it can reach the next ready-deque entry.

The three loop stages sum to the diagram's headline::

    handle_response 23.97 + build_response 50.65 + push_send 10.72 = 85.34 us

against ``trtllm-serve``'s 1.94 us of pure bookkeeping -- the 44.0x.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

_perf = time.perf_counter_ns


def spin(microseconds: float) -> None:
    """Burn ``microseconds`` of CPU **holding the GIL**.

    Models uninterruptible Python work. Do not replace with ``time.sleep``:
    sleeping drops the GIL and lets the loop run something else, which is the
    opposite of what these stages do.
    """
    if microseconds <= 0:
        return
    deadline = _perf() + int(microseconds * 1000)
    while _perf() < deadline:
        pass


def pad_to(start_ns: int, target_microseconds: float) -> None:
    """Spin until ``target_microseconds`` have elapsed since ``start_ns``.

    Stages do their real bookkeeping first and then call this, so the padding
    is only the difference between the real work (a couple of microseconds of
    dict building) and the measured cost. Attribution stays honest: if the real
    work ever grows past the target, this is a no-op rather than a rewind.
    """
    spin(target_microseconds - (_perf() - start_ns) / 1000.0)


@dataclass(frozen=True)
class Costs:
    """Per-stage costs in microseconds.

    ``ASYNCIO_GIL_PATH.md`` column names are given for each field.
    """

    # --- on the asyncio loop (dynamo, npw=0) -------------------------------
    #: GenerationResultBase._handle_response, run from _aresult_step -- i.e.
    #: ON the loop, not on the dispatch thread. Diagram: 23.97 us.
    handle_response_us: float = 23.97
    #: The dynamo worker's per-output response construction, inline because
    #: there are no postproc workers. Diagram: trtllm:build_response 50.65 us.
    build_response_us: float = 50.65
    #: ResponseSender.send -- pythonize + enqueue under the GIL we already
    #: hold. Push path only. Diagram: trtllm:push_send 10.72 us.
    push_send_us: float = 10.72

    # --- ingress, on the loop ---------------------------------------------
    #: The four pre-submit stages, each a measured p50 from the capture.
    #: Kept separate so the simulation emits the same NVTX range names and a
    #: capture of the simulation can be read back by ``capture_params``.
    normalize_request_us: float = 1.16
    setup_disagg_params_us: float = 37.95
    prepare_input_us: float = 1.93
    sampling_params_us: float = 17.42
    #: ``pybridge.invoke_generator`` (engine.rs:85) -- the per-request
    #: spawn_blocking crossing. Zero by default and deliberately so: the
    #: capture has no ``pybridge.*`` ranges (the Rust bridge NVTX was not
    #: armed), so there is no measurement to use, and the diagram's 1.05 ms for
    #: that box is latency waiting on a GIL the loop holds 98.7 % of the time
    #: rather than work. What the simulation models is the thread hop itself.
    invoke_generator_us: float = 0.0
    #: ``trtllm:engine_submit`` p50, measured on the capture.
    #:
    #: The diagram quotes 0.55 ms for "build request -> llm.generate_async",
    #: against a measured stage total of 213 us. The difference is not a
    #: contradiction: 0.55 ms is WALL from handler entry to submit returning,
    #: and it includes the gaps where the loop was running something else.
    #: Only the GIL-holding work belongs here -- the wall-clock gap has to
    #: emerge from the loop being busy, exactly as the iteration time does.
    engine_submit_us: float = 154.64

    # --- off the loop (Rust, tokio) ---------------------------------------
    #: Rust egress after the code change: chunk 6.56 + encode 3.31 +
    #: publish 1.69 = 11.56 us. No GIL on the real worker.
    rust_egress_us: float = 11.56
    #: Pull path only: depythonizing the yielded item plus the second and
    #: third cross-thread GIL acquisitions per response that push removes.
    pull_bridge_us: float = 11.56

    #: Multiply every stage. Useful to check that conclusions are structural
    #: rather than an artefact of one calibration.
    scale: float = 1.0

    def scaled(self, us: float) -> float:
        return us * self.scale

    @property
    def prepare_request_us(self) -> float:
        """Sum of the four pre-submit stages: 58.46 us on the capture."""
        return (
            self.normalize_request_us
            + self.setup_disagg_params_us
            + self.prepare_input_us
            + self.sampling_params_us
        )

    @property
    def loop_us_per_response_push(self) -> float:
        """85.34 us on the defaults -- the diagram's 3-stage total."""
        return self.scaled(
            self.handle_response_us + self.build_response_us + self.push_send_us
        )

    @property
    def loop_us_per_response_pull(self) -> float:
        """Pull keeps the same first two stages; the third moves to a tokio
        thread but costs a ready-deque entry per response instead."""
        return self.scaled(self.handle_response_us + self.build_response_us)

    def with_scale(self, scale: float) -> "Costs":
        return replace(self, scale=scale)


#: ``trtllm-serve``'s side of the same measurement, for reference in reports:
#: handle_response only, because formatting already happened in the 4
#: PostprocWorker processes.
SERVE_LOOP_US_PER_RESPONSE = 1.94

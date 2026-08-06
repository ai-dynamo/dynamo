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

import hashlib
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

_perf = time.perf_counter_ns

# ---------------------------------------------------------------------------
# Work-conservation ledger
# ---------------------------------------------------------------------------
#
# Every microsecond of modelled work is charged to the thread that burned it.
# An architecture is allowed to MOVE work off the loop, batch it, or amortise
# it; it is not allowed to quietly delete it and call the result a speedup.
# The benchmark prints this breakdown next to the throughput number so the two
# can be read together.
_LEDGERS: List[Tuple[str, List[float]]] = []
_ledger_local = threading.local()


def _cell() -> List[float]:
    cell = getattr(_ledger_local, "cell", None)
    if cell is None:
        cell = [0.0]
        _ledger_local.cell = cell
        # list.append is atomic under the GIL, so no lock is needed here.
        _LEDGERS.append((threading.current_thread().name, cell))
    return cell


_LEDGERS_OFFGIL: List[Tuple[str, List[float]]] = []


def _cell_offgil() -> List[float]:
    cell = getattr(_ledger_local, "offgil", None)
    if cell is None:
        cell = [0.0]
        _ledger_local.offgil = cell
        _LEDGERS_OFFGIL.append((threading.current_thread().name, cell))
    return cell


def offgil_ledger() -> Dict[str, float]:
    """Microseconds of work done WITHOUT the GIL, per thread."""
    totals: Dict[str, float] = {}
    for name, cell in list(_LEDGERS_OFFGIL):
        totals[name] = totals.get(name, 0.0) + cell[0]
    return {k: v for k, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v}


def reset_spin_ledger() -> None:
    del _LEDGERS[:]
    del _LEDGERS_OFFGIL[:]
    if hasattr(_ledger_local, "offgil"):
        del _ledger_local.offgil
    if hasattr(_ledger_local, "cell"):
        del _ledger_local.cell


def spin_ledger() -> Dict[str, float]:
    """Microseconds of modelled work, per thread name."""
    totals: Dict[str, float] = {}
    for name, cell in list(_LEDGERS):
        totals[name] = totals.get(name, 0.0) + cell[0]
    return {k: v for k, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v}


def spin(microseconds: float) -> None:
    """Burn ``microseconds`` of CPU **holding the GIL**.

    Models uninterruptible Python work. Do not replace with ``time.sleep``:
    sleeping drops the GIL and lets the loop run something else, which is the
    opposite of what these stages do.
    """
    if microseconds <= 0:
        return
    _cell()[0] += microseconds
    deadline = _perf() + int(microseconds * 1000)
    while _perf() < deadline:
        pass


# ---------------------------------------------------------------------------
# Off-GIL work
# ---------------------------------------------------------------------------
#
# `spin` models Python: CPU burned while HOLDING the GIL. Rust on a tokio
# thread is the opposite -- real CPU, no GIL -- and until now the simulator had
# no way to express it, which made "move this stage into Rust" unmeasurable and
# left process-offload looking like the only option.
#
# hashlib releases the GIL above HASHLIB_GIL_MINSIZE (2047 bytes), so hashing a
# large buffer is genuine CPU that other threads can run straight through.
# Verified: 1 -> 8 threads, wall 0.214 -> 0.223 s while total work grows 8x
# (perfect scaling); the same measurement with `spin` goes 0.061 -> 0.178 s.
# 32 KB: cache-resident, so this is compute-bound on the SHA unit rather than
# memory-bandwidth bound. A 1 MB buffer streamed from DRAM and shared bandwidth
# across cores, so pinned parallel runs interfered -- an off-GIL architecture
# measured 15 % slower in a 5-up batch while the baseline was unaffected.
_OFFGIL_BUF = b"\x5a" * (32 << 10)
#: hashlib only releases the GIL above HASHLIB_GIL_MINSIZE (2047 bytes).
_OFFGIL_MIN_BLOCK = 2048
#: Sliced as a memoryview, never as bytes. ``buf[:n]`` COPIES, and that memcpy
#: runs while HOLDING the GIL -- GIL-held work inside the one primitive whose
#: entire purpose is to not hold it. memoryview slicing is zero-copy.
_OFFGIL_MV = memoryview(_OFFGIL_BUF)
_offgil_bytes_per_us: Optional[float] = None


def _calibrate_offgil() -> float:
    """Bytes of SHA-256 per microsecond on this machine."""
    global _offgil_bytes_per_us
    if _offgil_bytes_per_us is None:
        probe, reps = 1 << 15, 256
        digest = hashlib.sha256()
        start = _perf()
        for _ in range(reps):
            digest.update(_OFFGIL_MV[:probe])
        elapsed_us = (_perf() - start) / 1000.0
        _offgil_bytes_per_us = (probe * reps) / max(elapsed_us, 1e-9)
    return _offgil_bytes_per_us


def spin_offgil(microseconds: float) -> None:
    """Burn ``microseconds`` of CPU **without** holding the GIL.

    What a Rust/tokio thread does. Charged to the ledger separately so a
    report can show how much work left the GIL entirely, as opposed to merely
    moving to another Python thread -- which the GIL serialises anyway, and
    which measured 0.13x when tried.
    """
    if microseconds <= 0:
        return
    _cell_offgil()[0] += microseconds
    # Size each block to land exactly on the deadline, then re-check. A fixed
    # ~10 us block overshot by up to 1.34x at the stage sizes that matter
    # (10-50 us), because the last block always ran to completion -- charging
    # off-GIL architectures a third more CPU than the model claimed, which
    # handicapped precisely the thing under test. `spin` is accurate to 1-3 %,
    # so the two were not comparable.
    bytes_per_us = _calibrate_offgil()
    deadline = _perf() + int(microseconds * 1000)
    digest = hashlib.sha256()
    while True:
        remaining_us = (deadline - _perf()) / 1000.0
        block = int(remaining_us * bytes_per_us)
        if block < _OFFGIL_MIN_BLOCK:
            return
        digest.update(_OFFGIL_MV[: min(block, len(_OFFGIL_MV))])


def pad_to(start_ns: int, target_microseconds: float) -> None:
    """Spin until ``target_microseconds`` have elapsed since ``start_ns``.

    Stages do their real bookkeeping first and then call this, so the padding
    is only the difference between the real work (a couple of microseconds of
    dict building) and the measured cost. Attribution stays honest: if the real
    work ever grows past the target, this is a no-op rather than a rewind.

    The ledger is charged the stage's FULL duration, not just the padding.
    Charging only the padding under-reports any architecture that does more
    real Python inside a padded stage -- and the effect is backwards: the
    baseline's ``ResponseSender.send`` does a cross-thread
    ``call_soon_threadsafe`` inside its padded range, so when that overruns
    10.72 us it is charged nothing, and an architecture that REMOVES that work
    looks like it added cost. Found by the batched-loop experiment.
    """
    elapsed_us = (_perf() - start_ns) / 1000.0
    remaining = target_microseconds - elapsed_us
    if remaining > 0:
        spin(remaining)
        _cell()[0] += elapsed_us  # the real work spin() did not see
    else:
        _cell()[0] += elapsed_us  # overran the model; charge what it cost


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

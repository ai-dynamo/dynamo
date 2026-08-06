# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A postprocessing pool that **dynamo owns**, with TRT-LLM left alone.

The one experiment that needs no TRT-LLM change
-----------------------------------------------
``postproc-procs`` (:mod:`egress_experiments.architectures.offloaded_postproc`)
is the fastest thing measured so far -- 3.07x -- but it buys that by turning
``num_postprocess_workers`` back on and registering dynamo's chunk builder as
TRT-LLM's postproc hook. That is a TRT-LLM config change plus an import-path
hook that has to be loaded *inside TRT-LLM's child process*, and it has to
un-revert ``main``'s dbeaa5b166, which added ``_strip_postprocess_workers``
(``components/src/dynamo/trtllm/workers/llm_worker.py:171-193``, called at
``:392``) precisely to delete that setting.

This module asks the other question: **what if dynamo spawns and owns the pool
itself?** ``num_postprocess_workers`` stays 0, ``proxy.py`` is untouched,
``base_worker.py`` is untouched, ``result.py`` is untouched. The only file that
changes is ``handler_base.py``.

The constraint that follows, and it is the whole point
------------------------------------------------------
With ``npw=0`` the proxy is unchanged, so responses still arrive on the
per-request ``AsyncQueue`` and ``async for res in generation_result`` still
drives ``GenerationResult._aresult_step`` (``result.py:1035``), which calls
``_handle_response`` (``result.py:454``) **on the event loop**. So::

    handle_response          23.97 us   ON THE LOOP -- unavoidable here
    trtllm:build_response    50.65 us   <- the only thing that can move
    trtllm:push_send         10.72 us   <- moves with it, see below

23.97 of the 85.34 us is spent before dynamo's handler sees ``res`` at all.
The ceiling is therefore 85.34 -> 23.97 + (whatever the hand-off costs), i.e.
about 2.8x if the hand-off is free -- and it is not free, which is what this
file measures rather than assumes.

``build_llm`` is deliberately **not** overridden: the proxy/dispatch path is
byte-for-byte ``baseline-push``. Only ``build_handler`` changes.

Shape
-----
::

    engine proc ── IPC ──▶ proxy_dispatch_result_thread ──put_nowait──▶ AsyncQueue
                                                                            │
                                              ┌── ON THE LOOP ──────────────┘
                                              │  _aresult_step -> handle_response  23.97
                                              │  slice the delta               O(delta)
                                              │  os.write(pipe)  ── fire and forget
                                              │  loop_meter.item()
                                              └────────────┬──────────────────┘
                                                           ▼
                        dyn-postproc worker PROCESS  (own interpreter, own GIL)
                            trtllm:build_response     50.65
                            encode + hand to Rust     10.72
                                                           │
                                                           ▼
                        one pipe the RUST side owns ──▶ demux by request id ──▶
                            the existing per-request mpsc.  The loop is never
                            woken to deliver the chunk.

Why the delta slice has to stay on the loop, and why it is cheap
----------------------------------------------------------------
``CompletionOutput.token_ids`` is CUMULATIVE (``result.py``, and
``handler_base.py:1188-1190``: *"The engine returns all tokens generated so far
for this choice. Calculate only the new tokens generated in this iteration"*).
The cumulative list lives in the app process, inside the ``GenerationResult``
that ``_handle_response`` just appended to. Shipping it to the worker would be
O(cumulative) per response -- the cost would grow all generation long, which is
the one thing that must not happen. So the loop slices
``output.token_ids[tokens_so_far:next_total_toks]`` -- O(delta), 40 ints at the
capture's ``stream_interval: 40``, one int in this benchmark -- and only the
delta crosses. The per-choice cursor ``output_tokens_per_choice``
(``handler_base.py:1185``) stays exactly where it is.

Everything else ``build_response`` needs is either in the frame or derivable:
``index``, ``finish_reason``, ``stop_reason``, ``res.finished``,
``num_input_tokens`` and ``sum(len(o.token_ids) for o in res.outputs)`` -- the
last two are O(1) per output and only read on the final chunk
(``handler_base.py:1234-1237``).

Where ``trtllm:push_send`` goes, and why
----------------------------------------
``push_send`` is ``ResponseSender.send``: ``decode_response`` (depythonize) plus
``tx.try_send`` (``push_egress.rs:204-229``). In the fire-and-forget design the
loop never holds the built chunk, so it cannot be the thing that hands it to
Rust -- the worker does, by writing to the socket Rust owns. The 10.72 us is
therefore charged **in the worker**, around the encode and the write. It is
moved, not deleted; the round-trip variant charges it back on the loop, in
``ResponseSender.send``, exactly as the baseline does.

Total modelled work per response is the same 96.90 us in all three cases::

    baseline-push          loop 85.34                          tokio 11.56
    dyn-postproc           loop 23.97   worker 61.37           tokio 11.56
    dyn-postproc-roundtrip loop 34.69   worker 50.65           tokio 11.56

Fire-and-forget vs round trip -- the ablation that decides it
-------------------------------------------------------------
``loop-executor-1`` (``dispatch_thread_work.py``) took 87 % of the loop's
modelled work away and came out at **0.77x**, slower than baseline. Its
structural counter says why: ``deque entries/item`` 1.043 against baseline's
0.019. Awaiting an off-loop result costs one ``call_soon_threadsafe`` per
response to hand it back -- which is precisely the per-response ready-deque
entry ``push_egress.rs`` exists to remove.

So the design here is fire-and-forget: the loop writes and moves on, and the
worker owns the egress. ``dyn-postproc-roundtrip`` is the same architecture with
the return arrow put back -- the loop awaits the built chunk and ``send``s it --
so the claim is measured, not asserted. It is registered as an architecture
rather than described in a comment for exactly that reason.

The reply variants
------------------
``dyn-postproc``      the worker's reply crosses back as opaque **bytes**. The
                      reader parses a fixed 22-byte header to get the request id
                      and passes the payload through without ever building a
                      Python object from it -- which is what Rust would do
                      (``push_egress.rs:402-412``: the tokio side only does
                      ``rx.recv().await``).
``dyn-postproc-dict`` identical, except the reader ``pickle.loads`` the payload
                      into a dict before delivering. Isolates one variable: what
                      the app process pays to *pythonize* the reply.
``dyn-postproc-rt``   the round trip. Same decode as ``-dict``, plus the loop
                      wake, plus ``push_send`` back on the loop.

WORK CONSERVATION -- read this
------------------------------
``costs.spin_ledger()`` only sees threads of *this* process, so the 61.37 us/item
that moves into the worker processes **disappears from the benchmark's "all
us/item" column**. It has not been deleted. Each worker reports its own
``spin_ledger()`` total back on the result pipe every
:data:`STATS_EVERY` items (and once more on the way out, so an aborted run still
accounts), and :meth:`DynPostproc.extra_report` prints
``offloaded_us_per_item``. Measured, medians, us/item::

    architecture             in-process   child-reported   total
    baseline-push                102.06             0.00  102.06
    postproc-procs                36.92            74.62  111.54
    dyn-postproc                  29.85            61.38   91.23
    dyn-postproc-nodrop           29.04            61.38   90.42
    dyn-postproc-roundtrip        71.70            50.65  122.35

``offloaded_us_per_item`` lands on 61.38 and 50.65 -- the modelled figures to
0.02 % -- so nothing was dropped. The totals spread 90-122 because ``pad_to``
charges the FULL elapsed time when a stage overruns its model, and the more
padded stages an architecture leaves on a contended thread the more it overruns:
``baseline-push``'s loop is charged 99.00 for a modelled 85.34, and the round
trip's 71.70 for a modelled 34.69 + ingress, because its loop is interrupted by
its own result callbacks. Fire-and-forget overruns least because it leaves the
fewest padded stages where the contention is.

``extra_report`` also prints the two numbers this experiment exists to produce:

``loop_dispatch_us_per_item``   wall microseconds the loop spends on the delta
                                slice + frame pack + write, measured with
                                ``perf_counter_ns`` on the loop itself. This is
                                REAL time, not modelled, so it does not appear
                                in the ledger. If it is not well under 50.65,
                                the architecture has not bought anything.
``hop_us_p50``/``p90``          full round-trip latency loop -> worker -> reader,
                                measured entirely on the parent's clock (the
                                loop's timestamp rides in the request frame and
                                is echoed back in the reply), so it needs no
                                assumption about cross-process clocks.

Read ``hop_us`` under saturation as a queue depth, not a latency: the benchmark
deliberately offers 2.2x what the loop can drain, so the pipes are full by
construction. Below saturation (``--batch 120``, 12,000 responses/s offered,
three runs each) it is 1.62-1.69 ms for ``dyn-postproc`` and 3.20-3.26 ms for
``-nodrop``, and even that is not the hop -- it is burst serialisation. One
engine iteration arrives as ONE IPC message (``base_worker.py:1252``,
``handle_for_ipc_batched``), so the loop fires ``batch/N`` frames at each worker
back to back and the last of 30 waits 30 x 61.37 us behind the first. At the
capture's geometry the same arithmetic gives 197.2 responses per 52.1 ms
iteration, i.e. 12.1 ms of postproc work per iteration against a 52.1 ms
iteration -- fits in one worker with room to spare.

Teardown
--------
``offloaded_postproc.py`` documents the deadlock this shape invites, and the
same rule applies: the reader thread must outlive the workers. If it is stopped
first, a worker blocked writing into a full result pipe never exits and
``join()`` hangs. So :meth:`_Pool.stop` closes the request pipes (workers see
EOF), joins the workers, and only then joins the reader -- which ends by itself,
because the parent dropped its own copy of the result write end at startup and
the last worker to exit drops the last one.

The one race worth naming: ``drive_push_egress`` calls ``response_sender.close()``
as soon as the handler's generator ends, but in fire-and-forget mode the final
chunk is still in flight through the worker. Closing first would drop it. The
handler therefore waits on a per-request ``asyncio.Event`` that the reader sets
after delivering the final chunk -- one ready-deque entry per REQUEST, not per
response, and zero of them in this benchmark, whose requests never finish. In
the real system this is what Rust already does for free: the mpsc stays open
until the worker's end-of-stream frame for that request arrives.

What would change in the real worker
------------------------------------
Only ``components/src/dynamo/trtllm/request_handlers/handler_base.py`` and one
new Rust ingress point. Nothing in ``tensorrt_llm``.

1. ``handler_base.py:1179-1278``. The ``for output in res.outputs:`` body --
   everything inside ``_nvtx.annotate("trtllm:build_response")`` -- moves into a
   pool worker. What stays on the loop is the three lines above it
   (``output_idx``, ``tokens_so_far``, ``next_total_toks``), the delta slice, and
   a write. The ``yield out`` at ``:1277`` goes away in fire-and-forget mode,
   which also means ``push_egress.py``'s ``async for response in stream``
   (``push_egress.py:187-196``) iterates an empty stream and only issues
   ``close()`` -- so ``drive_push_egress`` is untouched and still correct.
   ``_extract_logprobs`` (``:1199``) and ``_encode_and_pack_disaggregated_params``
   (``:1214``) would have to travel too; logprobs are already sliced with the
   same cursor, and the disagg-params branch is PREFILL-only, i.e. not on the
   decode path this is about.

2. **The worker needs a socket Rust owns.** Today ``ResponseSender``
   (``push_egress.rs:341-375``) is a ``#[pyclass]`` holding one end of a
   ``tokio::sync::mpsc`` (``push_egress.rs:394``), and the only way into it is
   ``send()`` under the GIL. What this design needs is a second, GIL-free door
   into the same channel: a per-worker unix socket, created by Rust at startup,
   with one tokio task per socket doing framed reads and demuxing by request id
   into the *existing* per-request ``mpsc::Sender``. Concretely, alongside
   ``response_channel`` (``push_egress.rs:385``), a registry
   ``HashMap<RequestId, mpsc::Sender<Annotated<Resp>>>`` populated in
   ``PythonPushEngine::generate`` (``push_egress.rs:464``) where the channel is
   already built, and a reader task that does
   ``tx.try_send(Annotated::from_data(serde_json::from_slice(payload)?))``.
   That task never touches Python, so ``py.allow_threads`` and the whole
   GIL-discipline argument in ``push_egress.rs:231-247`` do not apply to it at
   all. The Python handler would receive the worker-pool file descriptors the
   same way it receives the sender today -- on the ``Context``
   (``push_egress.rs:479-493``) -- except once per worker at startup rather than
   once per request.

3. **Nothing in TRT-LLM.** ``num_postprocess_workers`` stays 0, so
   ``proxy.py:457-464`` keeps ``zmq.PAIR``, ``base_worker.py:1252``
   (``handle_for_ipc_batched``) keeps shipping one message per iteration, and
   ``dbeaa5b166``'s ``_strip_postprocess_workers`` stays exactly as it is.

Measured
--------
Medians, one fresh process per measurement, serial on an idle 24-core box, rung
pinned with ``--batch 600``. n=5 for the headline rows, n=3 for the sweep::

    architecture             n   items/s  vs base  vs pp   loop us  deque/item
    baseline-push            5    10,053    1.00x   0.32x    99.00       0.013
    postproc-procs           5    31,269    3.11x   1.00x    25.99       0.012
    dyn-postproc             5    26,885    2.67x   0.86x    25.18       0.014
    dyn-postproc-nodrop      5    27,334    2.72x   0.87x    25.64       0.015
    dyn-postproc-dict        3    25,414    2.53x   0.81x    25.01       0.012
    dyn-postproc-roundtrip   5    11,676    1.16x   0.37x    60.40       1.143
    dyn-postproc-1           3    21,742    2.16x   0.70x    25.59       0.020
    dyn-postproc-2           3    25,798    2.57x   0.83x    25.14       0.014
    dyn-postproc-8           3    24,447    2.43x   0.78x    24.90       0.011

**The fire-and-forget claim holds, and by a factor of 2.3.** Same pool, same
worker code, same measured 50.65 us moved off the loop -- the ONLY difference is
whether the loop parks on the result. Fire-and-forget 26,885 against the round
trip's 11,676, and the benchmark's structural counter says why in one column:
``deque entries/item`` 0.014 against 1.143. That is the same mechanism
``loop-executor-1`` lost 23 % to, and it is worth more here than all 50.65 us of
``build_response``: the round trip moves the same work off the loop and still
only reaches 1.16x, because it hands the per-response ready-deque entry back.

What the hand-off actually costs, which was the open question::

    architecture             loop dispatch us/item   what the loop does
    dyn-postproc                            12.083   os.write to a pipe
    dyn-postproc-nodrop                      0.746   SimpleQueue.put
    dyn-postproc-dict                       13.716   os.write to a pipe
    dyn-postproc-roundtrip                  16.279   os.write + create_future

Both are far under the 50.65 us they replace, so the answer is yes. But 12.08
is twelve times the 0.95 us an isolated pipe write costs, and the whole
difference is that ``os.write`` releases the GIL -- see
:class:`DynPostprocNoDrop`, which gets the same throughput for 0.75 us of loop
time. The real change would be on the 0.75 side, because a pyo3 write does not
``allow_threads``.

Pool size: capacity is ``N * 1e6/61.37`` items/s, so N=1 gives 16,295 against a
loop that wants ~27,000 -- and 21,742 measured is the loop running into it
(``loop_stall_us_per_item`` 4.42, ``hop_us_p50`` 981 ms, 23,739 chunks stranded
in the pipes at teardown). N=2 covers it at 32,590; N=4 is the best measured; N=8
is *worse*, because eight more processes and eight more pipes cost the parent
more than the headroom is worth. The knee is at 2-4, and this is a firehose:
at the capture's actual 3,841 responses/s the pool needs 0.24 of one worker.

Pre-encoded bytes vs a dict costs 5 %: 26,885 against ``-dict``'s 25,414. That
is one ``pickle.loads`` per chunk on a thread that is not the loop, and it is
the smallest of the three effects measured here.

Reproduce with::

    python3 -m egress_experiments.bench --batch 600 --architecture dyn-postproc

one architecture per process, serially, on the whole box.

Deviations, and which way each one biases the result
----------------------------------------------------
1. **The reader thread is Python and holds the GIL.** In the real system it is
   a tokio task that never acquires the GIL. Here it parses a header and calls
   ``call_soon_threadsafe`` on the tokio-side loop for every chunk, all of it
   under the same GIL as the event loop. Pessimistic for this architecture.
2. **``pickle`` stands in for whatever Rust and the worker would agree on.**
   The encode cost is inside the padded ``push_send`` range in the worker, so
   it is charged; the *decode* is only paid in the ``-dict`` and ``-roundtrip``
   variants, which is what makes them the controls.
3. **``handle_response``'s 23.97 us is charged in full on the loop**, correctly:
   with ``npw=0`` there is no version of this design that avoids it.
"""

from __future__ import annotations

import array
import asyncio
import itertools
import multiprocessing
import os
import pickle
import queue
import select
import struct
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from egress_experiments import architectures, loop_meter
from egress_experiments.costs import Costs, pad_to, reset_spin_ledger, spin, spin_ledger
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import Driver
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------
#
# Fixed-size headers, little-endian, unaligned. Deliberately not pickle on the
# request leg: the loop pays for that one, and `struct.pack` + `array.tobytes`
# measured 0.13 us against pickle's 0.20 for the same payload. Both are noise
# next to the write syscall (~0.9 us), but the loop is the scarce resource and
# the honest thing is to make its half as cheap as it can be.
#
# total_len, client_id, seq, index, finish_code, stop_code, flags,
# prompt_tokens, total_completion_tokens, n_tokens, loop_write_ns
REQ_HDR = struct.Struct("<IIiIBBBIIIQ")
# total_len, kind, client_id, seq, final, loop_write_ns
RES_HDR = struct.Struct("<IBIiBQ")
#: Payload of a ``kind=1`` stats frame: worker id, spin us, items done.
STATS = struct.Struct("<idq")

_KIND_CHUNK = 0
_KIND_STATS = 1

_FLAG_FINISHED = 1

#: ``finish_reason`` as a code, so the hot frame stays fixed-size. The real
#: change would carry TRT-LLM's own enum; the set is small and closed either way.
_FINISH_CODES: Tuple[Optional[str], ...] = (None, "length", "stop", "unknown", "error")
_FINISH_INDEX: Dict[Optional[str], int] = {
    name: i for i, name in enumerate(_FINISH_CODES)
}

#: How often a worker reports its ledger back. 1/256 of a small frame is far
#: below the noise floor, and reporting continuously means an aborted run still
#: accounts for the work its children did.
STATS_EVERY = 256

#: Handoff mode only: how far the loop may run ahead of the feeder thread.
#: Bounded on purpose -- an unbounded producer hides a pool that cannot keep up,
#: which is the failure ``dispatch_thread_work.py`` documents at length. Over
#: the bound the loop does the write itself, so backpressure shows up as loop
#: cost rather than as memory.
OUTBOX_DEPTH = 8192

#: 1 MiB request pipes. The default 64 KiB holds ~1,500 frames; at the offered
#: rates here a worker that hiccups for a millisecond would then stall the LOOP,
#: which would be measuring the pipe rather than the architecture.
_PIPE_SZ = 1 << 20
_F_SETPIPE_SZ = 1031


def _grow_pipe(fd: int) -> None:
    try:
        import fcntl

        fcntl.fcntl(fd, _F_SETPIPE_SZ, _PIPE_SZ)
    except Exception:  # pragma: no cover - kernel/permission dependent
        pass


def _fanout(pairs: List[Tuple[Any, Any]]) -> None:
    """Runs on the TOKIO-side loop: N enqueues after one readiness event.

    Rust equivalent: the reader task's ``tx.try_send`` loop
    (``push_egress.rs:221``) after a single ``read()``. Nothing here touches
    the event loop the benchmark measures.
    """
    for sink, obj in pairs:
        sink.put_nowait(obj)


def _write_all(fd: int, payload: bytes) -> None:
    """Blocking, partial-write-safe. Used by threads allowed to block -- the
    workers and the handoff feeder -- and never by the event loop."""
    view = memoryview(payload)
    sent = 0
    while sent < len(view):
        sent += os.write(fd, view[sent:])


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------


def _worker_main(
    req_fd: int,
    res_fd: int,
    close_fds: List[int],
    costs: Costs,
    worker_id: int,
    charge_push_send: bool,
) -> None:
    """One dynamo-owned postproc process. Its own interpreter, its own GIL.

    Runs ``trtllm:build_response`` (``handler_base.py:1183-1266``) for one
    response at a time and writes the finished chunk to the result pipe -- the
    one the Rust side owns. It never talks back to the event loop.
    """
    # Every fd this child does not own: the other workers' request pipes, the
    # parent's read end of the result pipe. Leaving them open would keep the
    # write ends alive and no one would ever see EOF.
    for fd in close_fds:
        try:
            os.close(fd)
        except OSError:
            pass

    # The fork inherits the parent's ledger. From here this process owns its
    # own, so the total it reports back is exactly the work IT did.
    reset_spin_ledger()

    build_us = costs.scaled(costs.build_response_us)
    send_us = costs.scaled(costs.push_send_us)
    hdr_size = REQ_HDR.size
    res_hdr_size = RES_HDR.size

    buf = bytearray()
    pos = 0
    items = 0

    def _stats() -> None:
        payload = STATS.pack(worker_id, sum(spin_ledger().values()), items)
        try:
            _write_all(
                res_fd,
                RES_HDR.pack(res_hdr_size + len(payload), _KIND_STATS, 0, -1, 0, 0)
                + payload,
            )
        except OSError:
            pass

    try:
        while True:
            data = os.read(req_fd, 1 << 16)
            if not data:
                break
            buf += data
            while len(buf) - pos >= 4:
                (total,) = struct.unpack_from("<I", buf, pos)
                if len(buf) - pos < total:
                    break
                (
                    _,
                    client_id,
                    seq,
                    index,
                    finish_code,
                    stop_code,
                    flags,
                    prompt_tokens,
                    completion_tokens,
                    n_tokens,
                    write_ns,
                ) = REQ_HDR.unpack_from(buf, pos)
                tokens = array.array("i")
                if n_tokens:
                    tokens.frombytes(buf[pos + hdr_size : pos + total])
                pos += total

                finished = bool(flags & _FLAG_FINISHED)

                # ---- trtllm:build_response (handler_base.py:1183) ----------
                # Same construction, same order, same padded cost as
                # dynamo_sim/worker.py -- in another process.
                with range_("trtllm:build_response", color="yellow"):
                    start = _perf()

                    out: Dict[str, Any] = {
                        "token_ids": tokens.tolist(),
                        "index": index,
                    }
                    if finish_code:
                        out["finish_reason"] = _FINISH_CODES[finish_code]
                    if stop_code:
                        out["stop_reason"] = _FINISH_CODES[stop_code]

                    if out.get("finish_reason") or finished:
                        if not out.get("finish_reason"):
                            out["finish_reason"] = "unknown"
                        out["completion_usage"] = {
                            "prompt_tokens": int(prompt_tokens),
                            "completion_tokens": int(completion_tokens),
                            "total_tokens": int(prompt_tokens + completion_tokens),
                            "prompt_tokens_details": None,
                        }

                    pad_to(start, build_us)

                # ---- the hand-off to Rust ----------------------------------
                # trtllm:push_send, in the worker rather than on the loop: the
                # encode plus the enqueue (push_egress.rs:204-229), except the
                # enqueue is a write to the socket Rust reads instead of a
                # `tx.try_send` under the loop's GIL. Charged only when the loop
                # is NOT doing it -- see the round-trip variant.
                start = _perf()
                payload = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
                frame = (
                    RES_HDR.pack(
                        res_hdr_size + len(payload),
                        _KIND_CHUNK,
                        client_id,
                        seq,
                        1 if finished else 0,
                        write_ns,
                    )
                    + payload
                )
                if charge_push_send:
                    pad_to(start, send_us)
                # The write is OUTSIDE the padded range on purpose. Rust draws
                # the same line: `pybridge.push_send` covers depythonize +
                # `try_send` (push_egress.rs:209-229) and a full channel is a
                # SEPARATE range, `pybridge.push_blocked` (push_egress.rs:248).
                # Charging a blocked write as modelled work would let downstream
                # backpressure inflate this architecture's ledger -- measured at
                # 167 us/item against a modelled 61.37 before this was split.
                _write_all(res_fd, frame)

                items += 1
                if items % STATS_EVERY == 0:
                    _stats()

            if pos:
                del buf[:pos]
                pos = 0
    except (OSError, EOFError, BrokenPipeError):
        pass
    finally:
        _stats()
        for fd in (req_fd, res_fd):
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# The pool, as the app process sees it
# ---------------------------------------------------------------------------


class _Route:
    """Where a request's chunks go once they come back out of the pool.

    Fire-and-forget: straight onto the tokio-side sink, i.e. the Rust mpsc.
    The event loop is not involved and is not woken.
    """

    __slots__ = ("tokio_loop", "sink", "done")

    def __init__(self, tokio_loop: Any, sink: Any, done: Any) -> None:
        self.tokio_loop = tokio_loop
        self.sink = sink
        self.done = done


class _Pool:
    """N dynamo-owned worker processes, one reader thread, and the pipes.

    The reader thread stands in for the tokio task that would own the result
    socket. It is the one deviation that matters (see deviation 1 in the module
    docstring): here it is Python and holds the GIL.
    """

    def __init__(
        self,
        costs: Costs,
        workers: int,
        *,
        roundtrip: bool,
        decode_reply: bool,
        handoff: bool = False,
    ) -> None:
        self.costs = costs
        self.workers = max(1, workers)
        self.roundtrip = roundtrip
        self.decode_reply = decode_reply or roundtrip
        self.handoff = handoff

        self._req_w: List[int] = []
        self._res_r: int = -1
        self._procs: List[Any] = []
        self._reader: Optional[threading.Thread] = None
        self._feeder: Optional[threading.Thread] = None
        self._outbox: Optional["queue.SimpleQueue"] = None
        self._outbox_puts = 0
        #: Handoff mode: frames the loop had to write itself because the feeder
        #: was more than :data:`OUTBOX_DEPTH` behind.
        self.outbox_overflow = 0
        self._closing = False

        self.py_loop: Optional[asyncio.AbstractEventLoop] = None

        #: client_id -> route. Written on the loop, read on the reader thread;
        #: dict get/set are atomic under the GIL, so no lock on the hot path.
        self.routes: Dict[int, _Route] = {}
        #: Round-trip only: seq -> future the loop is parked on.
        self.pending: Dict[int, asyncio.Future] = {}
        self._seq = itertools.count(1)

        # -- counters ------------------------------------------------------
        #: Wall nanoseconds the LOOP spent slicing + packing + writing.
        #: Incremented per item rather than accumulated per request and flushed
        #: at the end: an aborted saturation run cancels every handler while the
        #: loop is blocked in teardown, so their `finally` blocks never run and
        #: a per-request flush loses most of the run.
        self.dispatch_ns = 0
        self.dispatched = 0
        #: Of :attr:`dispatch_ns`, the part spent waiting for a full pipe.
        #: Loop time either way, but it is backpressure and not the hand-off,
        #: so it is reported apart.
        self.stall_ns = 0
        #: EAGAIN on the loop's write. Must be ~0 or the number is about the
        #: pipe, not the architecture.
        self.write_stalls = 0
        #: Writes that failed because teardown closed the pipe first.
        self.write_failed = 0
        self.chunks_returned = 0
        #: Chunks whose request had already been unregistered.
        self.unrouted = 0
        #: Per-worker ``spin_ledger()`` totals -- work the parent CANNOT see.
        self.child_spin_us: Dict[int, float] = {}
        self.child_items: Dict[int, int] = {}
        #: Sampled loop -> worker -> reader round-trip, nanoseconds.
        self.hop_ns: List[int] = []
        self._hop_every = 32
        self.reader_errors: List[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self, py_loop: asyncio.AbstractEventLoop) -> None:
        self.py_loop = py_loop

        res_r, res_w = os.pipe()
        _grow_pipe(res_w)
        req_pipes = []
        for _ in range(self.workers):
            r, w = os.pipe()
            _grow_pipe(w)
            req_pipes.append((r, w))

        # Every pipe exists before the first fork, so each child can be told
        # exactly which fds are not its own. Doing it the other way round --
        # fork, then create the next pipe -- means child i inherits nothing of
        # child i+1 but child i+1 inherits child i's write end, and then child i
        # never sees EOF.
        ctx = multiprocessing.get_context("fork")
        for wid, (r, w) in enumerate(req_pipes):
            close_fds = [res_r] + [fd for pair in req_pipes for fd in pair]
            close_fds = [fd for fd in close_fds if fd not in (r, res_w)]
            proc = ctx.Process(
                target=_worker_main,
                args=(
                    r,
                    res_w,
                    close_fds,
                    self.costs,
                    wid,
                    not self.roundtrip,
                ),
                name=f"dyn_postproc_{wid}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)

        for r, w in req_pipes:
            os.close(r)
            # Non-blocking: the loop must never sit in the kernel waiting for a
            # worker. An EAGAIN is counted and then waited for explicitly, so a
            # stall is visible in the report rather than hidden in the wall.
            # In handoff mode the writer is a feeder thread that is allowed to
            # block, so blocking is both cheaper and the correct backpressure.
            os.set_blocking(w, self.handoff)
            self._req_w.append(w)
        # The parent's copy of the write end goes now, so the reader's read()
        # returns EOF once the last worker exits and not before.
        os.close(res_w)
        self._res_r = res_r

        if self.handoff:
            self._outbox = queue.SimpleQueue()
            self._feeder = threading.Thread(
                target=self._feeder_loop, name="rust-ingress-feeder", daemon=True
            )
            self._feeder.start()

        self._reader = threading.Thread(
            target=self._reader_loop, name="rust-egress-reader", daemon=True
        )
        self._reader.start()

    def stop(self) -> None:
        """Teardown order is load-bearing; see the module docstring."""
        self._closing = True
        if self._outbox is not None:
            try:
                self._outbox.put(None)
            except Exception:
                pass
        if self._feeder is not None:
            self._feeder.join(timeout=3.0)
            self._feeder = None
        for w in self._req_w:
            try:
                os.close(w)
            except OSError:
                pass
        self._req_w = []

        for proc in self._procs:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
        self._procs = []

        # LAST. A worker blocked writing into a full result pipe only unblocks
        # because this thread is still draining it.
        if self._reader is not None:
            self._reader.join(timeout=5.0)
            self._reader = None
        if self._res_r >= 0:
            try:
                os.close(self._res_r)
            except OSError:
                pass
            self._res_r = -1

    # -- the loop side -----------------------------------------------------

    def register(self, client_id: int, route: _Route) -> None:
        self.routes[client_id] = route

    def unregister(self, client_id: int) -> None:
        self.routes.pop(client_id, None)

    def next_seq(self) -> int:
        return next(self._seq)

    def send(self, client_id: int, frame: bytes) -> bool:
        """Fire and forget. Called ON the loop, once per response.

        Sticky by ``client_id`` so one request is always handled by one worker
        -- the same reason ``base_worker.py:1434`` buckets by ``client_id % N``.
        """
        if self._closing or not self._req_w:
            self.write_failed += 1
            return False
        outbox = self._outbox
        if outbox is not None:
            # `os.write` RELEASES the GIL, and re-acquiring it is what the loop
            # actually pays -- see :class:`DynPostprocNoDrop`. `SimpleQueue.put`
            # is a C-level append that never releases it (measured 0.07 us idle,
            # 1.10 us against three GIL contenders, against `os.write`'s 0.95 and
            # 204.5), so this models the hand-off Rust would do inline under the
            # GIL the handler already holds.
            if outbox.qsize() > OUTBOX_DEPTH:
                # The feeder is behind. Wait for it rather than writing to the
                # same fd from a second thread -- a batched feeder write is far
                # larger than PIPE_BUF, so two writers could interleave and
                # corrupt the frame stream. Loop time either way; counted.
                self.outbox_overflow += 1
                stall = _perf()
                while outbox.qsize() > OUTBOX_DEPTH and not self._closing:
                    time.sleep(0)
                self.stall_ns += _perf() - stall
            outbox.put((client_id, frame))
            return True
        fd = self._req_w[client_id % self.workers]
        try:
            os.write(fd, frame)
            return True
        except BlockingIOError:
            pass
        except OSError:
            self.write_failed += 1
            return False
        # Slow path: the pool is behind. Frames are far below PIPE_BUF, so a
        # pipe write is all-or-nothing and there is never a partial frame to
        # resume from -- only a full pipe to wait on.
        self.write_stalls += 1
        stall = _perf()
        try:
            while True:
                # Wait for writability rather than spinning: select drops the
                # GIL, a busy loop would not, and holding it here would freeze
                # the very worker that has to drain the pipe. Short timeout so a
                # dying pool costs microseconds of loop time, not milliseconds.
                select.select([], [fd], [], 0.002)
                try:
                    os.write(fd, frame)
                    return True
                except BlockingIOError:
                    if self._closing:
                        self.write_failed += 1
                        return False
                except OSError:
                    self.write_failed += 1
                    return False
        finally:
            self.stall_ns += _perf() - stall

    def _feeder_loop(self) -> None:
        """Handoff mode only: does the ``write(2)`` the loop refused to do.

        This thread exists to hold the GIL drop somewhere other than the event
        loop. The real change would not have it -- Rust would write inline,
        under the GIL the handler already holds, exactly as
        ``push_egress.rs:219-220`` argues for ``try_send``: *"there is no reason
        to drop the GIL for it, and dropping/reacquiring it would cost more than
        the send."* So read ``dyn-postproc-nodrop``'s LOOP cost as the estimate
        and its system cost as pessimistic by one thread.
        """
        outbox = self._outbox
        assert outbox is not None
        n = self.workers
        pending: List[List[bytes]] = [[] for _ in range(n)]
        stopping = False
        while True:
            item = outbox.get()
            if item is None:
                stopping = True
            else:
                pending[item[0] % n].append(item[1])
            # Drain whatever else has piled up, so one GIL round trip covers
            # many frames instead of one. Unbatched, this thread managed about
            # 5,000 frames/s -- it drops the GIL per write and pays ~200 us to
            # get it back -- and 84 % of the loop's hand-offs overflowed.
            while not stopping:
                try:
                    item = outbox.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                pending[item[0] % n].append(item[1])
            if not self._req_w:
                return
            try:
                for slot, frames in enumerate(pending):
                    if frames:
                        _write_all(self._req_w[slot], b"".join(frames))
                        del frames[:]
            except OSError:
                self.write_failed += 1
                return
            if stopping:
                return

    # -- the "Rust" side ---------------------------------------------------

    def _reader_loop(self) -> None:
        """One tokio task, modelled as a thread. Demuxes by request id.

        Reads framed replies off the single result pipe every worker writes to
        (frames are far below ``PIPE_BUF``, so concurrent writes from N
        processes are atomic and cannot interleave) and hands each chunk to the
        request's sink. In fire-and-forget mode that sink is the tokio-side
        queue -- the Rust mpsc -- and the event loop under study is never woken.

        Delivery is batched per ``read()``, and that is the faithful shape, not
        a shortcut: Rust's reader task gets ONE readiness event and then issues
        N ``tx.try_send`` calls with no scheduler hop between them
        (``push_egress.rs:221``). Doing one ``call_soon_threadsafe`` per chunk
        instead made this thread -- which in the real system is Rust and holds
        no GIL -- the bottleneck of the whole architecture, at which point the
        measurement is of the simulator's stand-in rather than of the design.
        """
        fd = self._res_r
        hdr = RES_HDR.size
        buf = bytearray()
        pos = 0
        seen = 0
        try:
            while True:
                data = os.read(fd, 1 << 16)
                if not data:
                    return
                buf += data
                batch: List[Tuple[Any, Any]] = []
                tokio_loop = None
                finals: List[Any] = []
                while len(buf) - pos >= 4:
                    (total,) = struct.unpack_from("<I", buf, pos)
                    if len(buf) - pos < total:
                        break
                    (
                        _,
                        kind,
                        client_id,
                        seq,
                        final,
                        write_ns,
                    ) = RES_HDR.unpack_from(buf, pos)
                    payload = bytes(buf[pos + hdr : pos + total])
                    pos += total

                    if kind == _KIND_STATS:
                        worker_id, spin_us, items = STATS.unpack(payload)
                        self.child_spin_us[worker_id] = spin_us
                        self.child_items[worker_id] = items
                        continue

                    seen += 1
                    if seen % self._hop_every == 0:
                        self.hop_ns.append(_perf() - write_ns)

                    obj: Any = pickle.loads(payload) if self.decode_reply else payload

                    if self.roundtrip:
                        # NOT batched, deliberately: the round trip's whole cost
                        # is that the loop is parked on a per-response future,
                        # so waking it is per response by construction. This is
                        # the same shape `loop-executor-1` pays 1.043 deque
                        # entries per item for.
                        self._resolve_on_loop(seq, obj)
                        continue

                    route = self.routes.get(client_id)
                    if route is None:
                        self.unrouted += 1
                        continue
                    self.chunks_returned += 1
                    tokio_loop = tokio_loop or route.tokio_loop
                    batch.append((route.sink, obj))
                    if final:
                        finals.append(route.done)

                if batch and tokio_loop is not None:
                    try:
                        tokio_loop.call_soon_threadsafe(_fanout, batch)
                    except RuntimeError:
                        # Loop already closed: teardown beat us to it.
                        self.unrouted += len(batch)
                for done in finals:
                    try:
                        self.py_loop.call_soon_threadsafe(done.set)
                    except RuntimeError:
                        pass
                if pos:
                    del buf[:pos]
                    pos = 0
        except (OSError, EOFError):
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.reader_errors.append(f"{type(exc).__name__}: {exc}")

    def _resolve_on_loop(self, seq: int, obj: Any) -> None:
        """Round trip only: hand the chunk BACK to the loop.

        One ``call_soon_threadsafe`` per response, i.e. one ready-deque entry
        per response -- the thing push egress exists to remove, put back on
        purpose so the fire-and-forget claim can be measured against it.
        """
        loop = self.py_loop
        if loop is None:
            return
        self.chunks_returned += 1
        try:
            loop.call_soon_threadsafe(self._set_result, seq, obj)
        except RuntimeError:
            self.unrouted += 1

    def _set_result(self, seq: int, obj: Any) -> None:
        future = self.pending.pop(seq, None)
        if future is not None and not future.done():
            future.set_result(obj)

    # -- reporting ---------------------------------------------------------

    @property
    def offloaded_us(self) -> float:
        return sum(self.child_spin_us.values())


# ---------------------------------------------------------------------------
# The handler -- the ONLY thing this architecture changes
# ---------------------------------------------------------------------------


class _DynPostprocHandlerBase(TrtllmWorkerHandler):
    """``handler_base._generate_locally_impl`` with the build hived off.

    Ingress is byte-for-byte ``dynamo_sim/worker.py``: the same four pre-submit
    stages under the same NVTX names, the same ``generate_async`` call, the same
    ``async for res in generation_result`` -- which means ``handle_response``
    still runs on the loop inside ``_aresult_step``, exactly as it does today.
    """

    def __init__(
        self,
        llm: FakeLLM,
        costs: Optional[Costs] = None,
        records: Optional[Dict[str, RequestRecord]] = None,
        *,
        pool: Optional[_Pool] = None,
    ) -> None:
        super().__init__(llm, costs=costs, records=records)
        assert pool is not None
        self.pool = pool

    # -- shared ingress ----------------------------------------------------

    def _submit(self, request: dict) -> Any:
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
        return self.llm.generate_async(
            inputs=request.get("token_ids"),
            sampling_params=sampling_params,
            disaggregated_params=None,
            streaming=True,
            trace_headers=None,
            scheduling_params=None,
            priority=0.5,
            cache_salt=None,
        )


class FireAndForgetHandler(_DynPostprocHandlerBase):
    """The loop writes and moves on. It never sees the built chunk.

    Yields nothing, so ``drive_push_egress``'s ``async for response in stream``
    (``push_egress.py:187``) has nothing to send and only issues ``close()`` --
    which is why ``trtllm:push_send`` is charged in the worker instead.
    """

    async def _generate_locally_impl(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        generation_result = self._submit(request)

        pool = self.pool
        client_id = generation_result.client_id
        num_input_tokens = len(request.get("token_ids") or [])

        # Rust hands the sender over on the context (push_egress.rs:479-493);
        # in the real change it would hand over the pool's descriptors the same
        # way. Here it is what gives the reader the request's mpsc.
        sender = getattr(context, "response_sender", None)
        done = asyncio.Event()
        pool.register(
            client_id,
            _Route(
                getattr(sender, "_tokio_loop", None),
                getattr(sender, "_sink", None),
                done,
            ),
        )

        output_tokens_per_choice: Dict[int, int] = {}
        finished = False

        try:
            # Unchanged from the baseline: __anext__ -> _aresult_step ->
            # aqueue.get() then _handle_response. The 23.97 us runs HERE.
            async for res in generation_result:
                finished = res.finished
                completion_tokens = (
                    sum(len(o.token_ids) for o in res.outputs) if finished else 0
                )
                for output in res.outputs:
                    with range_("trtllm:postproc_dispatch", color="yellow"):
                        start = _perf()

                        output_idx = getattr(output, "index", 0) or 0
                        tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                        next_total_toks = len(output.token_ids)

                        # O(delta). NEVER O(cumulative) -- token_ids grows for
                        # the whole generation (handler_base.py:1188-1190).
                        delta = output.token_ids[tokens_so_far:next_total_toks]
                        n_tokens = len(delta)
                        body = array.array("i", delta).tobytes() if n_tokens else b""

                        frame = (
                            REQ_HDR.pack(
                                REQ_HDR.size + len(body),
                                client_id,
                                -1,
                                output_idx,
                                _FINISH_INDEX.get(output.finish_reason, 0),
                                _FINISH_INDEX.get(output.stop_reason, 0),
                                _FLAG_FINISHED if finished else 0,
                                num_input_tokens,
                                completion_tokens,
                                n_tokens,
                                start,
                            )
                            + body
                        )
                        ok = pool.send(client_id, frame)

                        output_tokens_per_choice[output_idx] = next_total_toks
                        pool.dispatch_ns += _perf() - start
                        pool.dispatched += 1

                    if ok:
                        # The last point the loop touches this item.
                        self.responses_yielded += 1
                        loop_meter.item()
        finally:
            if finished:
                # Do not let drive_push_egress close the stream while the final
                # chunk is still inside the pool. One deque entry per REQUEST.
                try:
                    await asyncio.wait_for(done.wait(), timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            pool.unregister(client_id)

        if False:  # pragma: no cover - never runs; makes this an async generator
            yield {}


class RoundTripHandler(_DynPostprocHandlerBase):
    """The ablation: identical, except the loop awaits the chunk and sends it.

    One ``call_soon_threadsafe`` onto the event loop per response to hand the
    result back -- the ready-deque entry ``loop-executor-1`` paid 23 % for.
    """

    async def _generate_locally_impl(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        generation_result = self._submit(request)

        pool = self.pool
        loop = asyncio.get_running_loop()
        client_id = generation_result.client_id
        num_input_tokens = len(request.get("token_ids") or [])

        output_tokens_per_choice: Dict[int, int] = {}

        # No try/finally: this variant registers no route, and the loop-side
        # counters are incremented per item rather than flushed at the end --
        # an aborted saturation run cancels every handler while the loop is
        # blocked in teardown, so a `finally` would never run.
        async for res in generation_result:
            finished = res.finished
            completion_tokens = (
                sum(len(o.token_ids) for o in res.outputs) if finished else 0
            )
            for output in res.outputs:
                with range_("trtllm:postproc_dispatch", color="yellow"):
                    start = _perf()

                    output_idx = getattr(output, "index", 0) or 0
                    tokens_so_far = output_tokens_per_choice.get(output_idx, 0)
                    next_total_toks = len(output.token_ids)

                    delta = output.token_ids[tokens_so_far:next_total_toks]
                    n_tokens = len(delta)
                    body = array.array("i", delta).tobytes() if n_tokens else b""

                    seq = pool.next_seq()
                    future = loop.create_future()
                    pool.pending[seq] = future

                    frame = (
                        REQ_HDR.pack(
                            REQ_HDR.size + len(body),
                            client_id,
                            seq,
                            output_idx,
                            _FINISH_INDEX.get(output.finish_reason, 0),
                            _FINISH_INDEX.get(output.stop_reason, 0),
                            _FLAG_FINISHED if finished else 0,
                            num_input_tokens,
                            completion_tokens,
                            n_tokens,
                            start,
                        )
                        + body
                    )
                    ok = pool.send(client_id, frame)

                    output_tokens_per_choice[output_idx] = next_total_toks
                    pool.dispatch_ns += _perf() - start
                    pool.dispatched += 1

                if not ok:
                    pool.pending.pop(seq, None)
                    continue

                # THE difference. Parking here is what costs a
                # call_soon_threadsafe per response on the way back.
                out = await future
                self.responses_yielded += 1
                # -> drive_push_egress -> ResponseSender.send: push_send's
                # 10.72 us and loop_meter.item(), on the loop, as baseline.
                yield out


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class DynPostproc(architectures.Architecture):
    """Fire-and-forget: dynamo's own postproc pool, TRT-LLM untouched."""

    name = "dyn-postproc"
    description = "build_response in 4 DYNAMO-owned processes, fire-and-forget"
    egress = "push"

    workers = 4
    roundtrip = False
    decode_reply = False
    handoff = False

    def __init__(self) -> None:
        self._pool: Optional[_Pool] = None

    # build_llm is NOT overridden: the proxy/dispatch path is baseline-push,
    # and handle_response still runs on the loop. That is the constraint.

    def build_handler(
        self,
        llm: FakeLLM,
        costs: Costs,
        records: Dict[str, RequestRecord],
    ) -> Any:
        pool = _Pool(
            costs,
            self.workers,
            roundtrip=self.roundtrip,
            decode_reply=self.decode_reply,
            handoff=self.handoff,
        )
        pool.start(asyncio.get_running_loop())
        self._pool = pool
        handler_cls = RoundTripHandler if self.roundtrip else FireAndForgetHandler
        return handler_cls(llm, costs=costs, records=records, pool=pool)

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        if self._pool is not None:
            self._pool.stop()

    def extra_report(self) -> Dict[str, Any]:
        pool = self._pool
        if pool is None:
            return {}
        items = max(1, pool.dispatched)
        moved = pool.costs.scaled(
            pool.costs.build_response_us
            + (0.0 if self.roundtrip else pool.costs.push_send_us)
        )
        hops = sorted(pool.hop_ns)

        def _pct(q: float) -> float:
            if not hops:
                return 0.0
            return hops[min(len(hops) - 1, int(q * len(hops)))] / 1000.0

        report: Dict[str, Any] = {
            "pool_workers": pool.workers,
            "mode": "roundtrip" if self.roundtrip else "fire-and-forget",
            "reply": "dict" if pool.decode_reply else "bytes",
            "write": "feeder-thread" if pool.handoff else "loop (os.write)",
            "dispatched": pool.dispatched,
            "chunks_returned": pool.chunks_returned,
            # THE number this experiment exists to produce: real wall time the
            # loop spends on slice + pack + write, against build_response's
            # 50.65 us that it replaces. Net of pipe stalls, which are loop time
            # too but are backpressure rather than the hand-off.
            "loop_dispatch_us_per_item": round(
                (pool.dispatch_ns - pool.stall_ns) / items / 1000.0, 3
            ),
            "loop_stall_us_per_item": round(pool.stall_ns / items / 1000.0, 3),
            "hop_us_p50": round(_pct(0.50), 1),
            "hop_us_p90": round(_pct(0.90), 1),
            "offloaded_us_per_item": round(pool.offloaded_us / items, 2),
            "offloaded_us_total": round(pool.offloaded_us),
            "child_items": sum(pool.child_items.values()),
            "pool_capacity_items_per_s": round(pool.workers * 1e6 / moved)
            if moved
            else 0,
            "write_stalls": pool.write_stalls,
        }
        if pool.handoff:
            report["outbox_overflow"] = pool.outbox_overflow
        if pool.write_failed:
            report["write_failed_at_teardown"] = pool.write_failed
        if pool.unrouted:
            report["unrouted"] = pool.unrouted
        if pool.reader_errors:
            report["reader_errors"] = pool.reader_errors[:2]
        return report


class DynPostproc1(DynPostproc):
    name = "dyn-postproc-1"
    description = "build_response in 1 DYNAMO-owned process, fire-and-forget"
    workers = 1


class DynPostproc2(DynPostproc):
    name = "dyn-postproc-2"
    description = "build_response in 2 DYNAMO-owned processes, fire-and-forget"
    workers = 2


class DynPostproc8(DynPostproc):
    name = "dyn-postproc-8"
    description = "build_response in 8 DYNAMO-owned processes, fire-and-forget"
    workers = 8


class DynPostprocDict(DynPostproc):
    """Control: the reply is pythonized in the app process before delivery."""

    name = "dyn-postproc-dict"
    description = "fire-and-forget, but the reply is decoded to a dict in-process"
    decode_reply = True


class DynPostprocNoDrop(DynPostproc):
    """What the loop's hand-off costs when it does NOT drop the GIL.

    ``os.write`` releases the GIL (``Py_BEGIN_ALLOW_THREADS``), and re-acquiring
    it is what the loop actually pays. Per hand-off, against a variable number
    of threads holding the GIL in 25 us slices -- which is what the dispatch
    thread, the reader thread and the tokio-side loop do here::

        primitive                    idle      3 contenders
        os.write to a pipe          0.947 us      204.546 us
        queue.Queue.put             0.683 us       10.079 us
        queue.SimpleQueue.put       0.069 us        1.103 us
        collections.deque.append    0.044 us        0.346 us

    The syscall itself is 0.95 us. Everything above that is waiting to get the
    GIL back, and it is 200x. The real change would not pay it: the loop would
    call a ``#[pymethods]`` function doing a non-blocking ``write(2)``
    **without** ``allow_threads`` -- the same judgement ``push_egress.rs:219-220``
    already makes about ``try_send``: *"there is no reason to drop the GIL for
    it, and dropping/reacquiring it would cost more than the send."*

    Python cannot express "syscall without releasing the GIL", so this variant
    brackets it from the other side: the loop does a ``SimpleQueue.put``, which
    is a C-level append that never releases the GIL, and a feeder thread does
    the write. Read its LOOP cost as the estimate of the real one, and its system
    throughput as pessimistic -- the real design has no feeder thread, and this
    one is a fifth GIL contender.
    """

    name = "dyn-postproc-nodrop"
    description = "fire-and-forget, but the loop's hand-off never drops the GIL"
    handoff = True


class DynPostprocRoundTrip(DynPostproc):
    """The ablation the fire-and-forget claim is measured against."""

    name = "dyn-postproc-roundtrip"
    description = "same pool, but the LOOP awaits the chunk and sends it"
    roundtrip = True
    decode_reply = True


for _factory in (
    DynPostproc,
    DynPostproc1,
    DynPostproc2,
    DynPostproc8,
    DynPostprocDict,
    DynPostprocNoDrop,
    DynPostprocRoundTrip,
):
    architectures.register(_factory)

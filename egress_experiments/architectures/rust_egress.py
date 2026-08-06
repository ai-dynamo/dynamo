# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert to an owned Rust value at the FIRST step, not the last.

The conclusion this experiment tests
------------------------------------
Experiment 6 measured ``postproc-procs`` (4 processes) at 3.01x and
``postproc-threads`` (4 threads) at 0.13x and concluded: *only a process that
never held the Python object can do this work, because building a chunk means
reading Python objects, so tokio would have to take the GIL per response.*

That is too strong, and push egress itself is the counter-example.
``lib/bindings/python/rust/push_egress.rs`` does exactly the forbidden thing and
it works:

* ``TypedSink::send`` (push_egress.rs:204-254) runs under the GIL the Python
  handler **already holds** -- ``dynamo_nvtx_range!("pybridge.push_send")`` at
  :209 explicitly notes "there is no GIL *acquisition* inside this range";
* ``decode_response`` (push_egress.rs:284-312) ``depythonize``s into
  ``Annotated<Resp>``, an **owned Rust value**;
* ``tx.try_send(frame)`` (push_egress.rs:221) hands it to a bounded tokio mpsc
  whose consumer only does ``rx.recv().await`` (push_egress.rs:402-412) and
  never touches Python. The module docstring states the invariant at :29-30 --
  "The tokio side only ever does ``rx.recv().await``: it never acquires the GIL
  and never touches a Python object."

Contrast the PULL path: ``python_payload.rs:25`` is
``pub(crate) struct PythonPayload(Py<PyAny>)`` -- a live Python handle -- so
every later touch needs ``Python::with_gil`` (python_payload.rs:44, 57, 192).

So the rule is not "Rust cannot do this". The rule is:

    convert ONCE, under a GIL you already hold, into an owned Rust value;
    everything downstream is GIL-free.

Today that conversion is the LAST step. ``handle_response`` (23.97 us) and
``trtllm:build_response`` (50.65 us) both run in Python on the loop *first*, and
only then does ``push_send`` (10.72 us) convert the finished chunk. This module
moves the conversion to the FIRST step -- to
``proxy_dispatch_result_thread``, which already holds the GIL because it just
``pickle.loads``-ed the IPC message (``tensorrt_llm/executor/ipc.py:389``,
reached from ``proxy.py:535``) -- and then runs the other two stages in Rust,
on tokio threads, with no GIL at all.

The variants
------------
``rust-egress``       Python still owns the zmq socket. The dispatch thread
                      unpickles as it does today and ``depythonize``s each
                      response into an owned Rust struct **under the GIL it
                      already holds**. ``handle_response`` + ``build_response``
                      then run off-GIL on a pool of tokio threads. The loop
                      does one handle move per response and nothing else.
``rust-egress-full``  The endgame: Rust owns the IPC socket, frames arrive in a
                      language-neutral format, and a response is **never a
                      Python object at all**. Identical to ``rust-egress``
                      except the conversion is also off-GIL.
``rust-egress-gil``   THE ABLATION. Byte-for-byte the same topology, threads,
                      queues and hand-offs as ``rust-egress`` -- but the moved
                      74.62 us is charged with :func:`~egress_experiments.costs.spin`
                      instead of :func:`~egress_experiments.costs.spin_offgil`.
                      One line different. It isolates "no GIL" from "off the
                      loop", which is the confound in every earlier result.
``rust-egress-w1``    1 tokio thread; ``rust-egress-w8`` 8.
``rust-egress-full-w8``  8 tokio threads, fully off-GIL.
``rust-egress-bp``    ``rust-egress`` with the conversion *inside* the credit
                      window, so it happens exactly once per DELIVERED response.
                      See "the ungated converter" below; this is the bracket
                      that corresponds to a system with real backpressure.
``rust-egress-full-bp``  the same for the fully off-GIL variant.

The ungated converter, and why there are two brackets
-----------------------------------------------------
This benchmark works by offering ~2.4x what the loop can drain and measuring the
drain rate. That is the right method for a loop-bound design, but it punishes
any architecture whose *producer* side is ungated: the reader converts every
response the engine emits, including the 60 % the loop will never reach, so its
per-DELIVERED-item cost is inflated by exactly ``offered / delivered``.
:meth:`extra_report` prints that factor as ``converts_per_loop_item``.

It is a benchmark artefact, not a property of the design. In the real system the
tokio channel is bounded (``PUSH_CHANNEL_DEPTH = 128``, push_egress.rs:80) and a
full channel blocks the sender (push_egress.rs:249), which backs up the zmq lane
and stalls the engine -- so exactly one conversion happens per delivered
response. The simulator cannot reproduce that: gating the *reader* stops it
draining the IPC lane, which freezes ``responses_dispatched``, which is the
signal the benchmark uses to prove saturation. (Found by the dispatch-thread
experiment; ``dispatch_thread_work.py`` documents the same constraint.)

So both brackets are measured. ``rust-egress`` puts the conversion where the
real change puts it -- on the dispatch thread, ungated -- and pays the
artefact. ``rust-egress-bp`` moves it inside the credit window, one conversion
per delivered response, which is what a backpressured system does; the price is
that the conversion then runs on a pool thread rather than the reader. The GIL
cost is identical either way -- one Python thread holds the GIL for
``convert_us`` per response, and which thread it is does not change the total --
so ``rust-egress-bp`` is the honest steady-state number and ``rust-egress`` is
the pessimistic one.

Where the modelled work goes
----------------------------
Per response, against the baseline's 85.34 us of loop time::

    stage                     baseline            rust-egress          rust-egress-full
    ----------------------------------------------------------------------------------
    convert (depythonize)     10.72  loop, GIL    10.72 dispatch, GIL  10.72 reader, NO GIL
    handle_response           23.97  loop, GIL    23.97 tokio, NO GIL  23.97 tokio, NO GIL
    build_response            50.65  loop, GIL    50.65 tokio, NO GIL  50.65 tokio, NO GIL
    loop hand-off              0                  ~0                   ~0
    ----------------------------------------------------------------------------------
    GIL-holding per response  85.34               10.72                0

Nothing is deleted. ``costs.spin_ledger()`` shows the GIL half and
``costs.offgil_ledger()`` shows the other half; :meth:`extra_report` prints
both, per item, so the 74.62 us can be seen arriving in the off-GIL ledger.

Why the conversion costs ``push_send_us``
-----------------------------------------
10.72 us is the *measured* cost of exactly one ``depythonize`` + enqueue in this
codebase -- ``trtllm:push_send``, capture 355778 p50. It is the only measured
Python->Rust conversion available, so it is what the new conversion point is
charged. That is a modelling choice and it is the load-bearing one; override it
with ``DYN_SIM_CONVERT_US=<us>`` to see how sensitive the answer is. Note the
direction of the error: a ``tllm.Response`` state tuple is *smaller* than the
output chunk it produces (3 fields vs a dict with token_ids + usage), so if
anything 10.72 over-charges.

Measured
--------
**Every architecture on the SAME ladder rung.** Serial, solo, one fresh process
per run, pinned to 4 cores, ``--cost-scale 2`` so that all six saturate at batch
240 against the same 24,000/s offer (see trap B below). Medians of 3; spreads
were under 2 % except ``postproc-procs`` (7 %)::

    architecture           items/s  vs base   loop us   GIL us   off-GIL us
    ---------------------------------------------------------------------
    postproc-procs          13,993   2.75x     60.53    76.71   149.3 (procs)
    rust-egress-full-bp     12,507   2.46x      1.95    25.07   171.06
    rust-egress-bp          11,734   2.31x      2.28    46.91   149.70
    rust-egress              6,155   1.21x      6.69   110.16   150.77
    rust-egress-gil-bp       5,155   1.01x      8.69   213.92     0
    baseline-push            5,087   1.00x    197.17   201.26     0

At the campaign's standard ``--cost-scale 1``, same protocol, medians of 3 --
but note that each architecture lands on the rung where IT saturates, so the
offered load differs and the rows are not directly comparable::

    architecture           items/s  vs base   loop us   rung (offered/s)
    ---------------------------------------------------------------------
    postproc-procs          24,455   2.57x     43.05    600 (44,678)
    rust-egress-full-bp     18,834   1.98x      0.23    240 (22,205)
    rust-egress-bp          17,403   1.83x      0.34    240 (22,064)
    rust-egress             14,754   1.55x      0.63    240 (22,057)
    baseline-push            9,510   1.00x    101.85    240 (23,318)
    rust-egress-gil-bp       9,375   0.99x      1.90    240 (24,518)
    rust-egress-gil          7,876   0.83x      2.62    240 (24,876)

**The ablation, which is the point of the whole file.** ``rust-egress-bp`` and
``rust-egress-gil-bp`` are the same threads, the same queues, the same credit
window, the same hand-offs and the same total modelled microseconds. The only
difference is whether the moved 74.62 us is burned with ``spin_offgil`` or
``spin``::

    off the loop AND off the GIL   rust-egress-bp       11,734    2.31x
    off the loop, ON the GIL       rust-egress-gil-bp    5,155    1.01x
    on the loop (baseline)         baseline-push         5,087    1.00x

    "no GIL"       = 11,734 / 5,155 = 2.28x
    "off the loop" =  5,155 / 5,087 = 1.01x

So **100 % of the win is "no GIL" and 0 % is "off the loop"**, and that single
result explains both of the earlier ones. ``postproc-threads``'s 0.13x and
``postproc-procs``'s 3.01x were never about processes versus threads; they were
about the GIL. A process drops it, and so does a tokio thread running Rust. The
process was sufficient, not necessary -- and it costs a pickle round trip and
four extra OS processes to buy what a ``depythonize`` buys for one 10.72 us
conversion.

The loop number is the structural one. ``rust-egress-full-bp`` puts **1.95 us**
of modelled work per response on the event loop against the baseline's 197.17
and ``postproc-procs``'s 60.53 -- 100x and 31x reductions. That matters more
than the throughput column, because a loop cost of 60.53 us/item is a hard
ceiling at 1e6/60.53 = 16,500 items/s that no amount of hardware removes, while
1.95 us/item is not a ceiling at all. ``postproc-procs`` is measured *at* its
loop ceiling; ``rust-egress-*`` is measured at whatever the four pinned cores
allow.

Conservation holds. ``offgil_us_per_item`` lands at 171.21 against an expected
170.68 for ``rust-egress-full-bp`` and 149.84 against 149.24 for
``rust-egress-bp`` -- 0.3 %. Total work per item is 196.1 and 196.6 against the
baseline's 201.3, i.e. the 74.62 (or 85.34) us really did move rather than
vanish; ``postproc-procs`` comes out at 226.0, the extra being its relay and
second pickle hop.

Did it beat 3.01x? Not on four cores in this simulator -- 2.46x against
``postproc-procs``'s 2.75x on the shared rung. Three reasons, in order of size:

1. **They are limited by different things.** ``postproc-procs`` is loop-bound
   and sitting on its ceiling. ``rust-egress-*`` is core-bound: 171 us/item of
   off-GIL work over four pinned cores is 23,400 items/s at best, and the engine
   process and the loop need some of those cores too.
2. **``spin_offgil`` costs more real CPU than it charges** (trap C) and is
   memory-bandwidth bound where ``spin`` is register-only (trap A). Both
   handicap the off-GIL architectures against a ``spin``-based process pool for
   reasons that have nothing to do with either design.
3. **``rust-egress`` (ungated) pays the benchmark's oversubscription tax** --
   3.75 conversions per delivered item at scale 2. That is trap-shaped too: a
   real bounded channel makes it exactly 1, which is what ``-bp`` measures.

The pickle / nanobind boundary -- what actually has to change
-------------------------------------------------------------
``depythonize`` walks mappings and sequences. It cannot walk a nanobind class,
which exposes properties rather than a mapping protocol. So ``rust-egress`` is
not a pure Rust-side change; something has to hand it a walkable structure.
Three facts settle what that costs:

1. **The C++ objects already define pickle state.**
   ``cpp/tensorrt_llm/nanobind/executor/request.cpp:1008-1009``::

       auto responseGetstate = [](tle::Response const& self)
       { return nb::make_tuple(self.getRequestId(), self.getResult(), self.getClientId()); };

   registered at ``request.cpp:1049-1050``, and ``Result`` at
   ``request.cpp:971-976`` returns a 14-field tuple
   (``isFinal, outputTokenIds, cumLogProbs, logProbs, contextLogits,
   generationLogits, encoderOutput, finishReasons, sequenceIndex,
   isSequenceFinal, decodingIter, avgDecodedTokensPerIter, contextPhaseParams,
   requestPerfMetrics``), registered at ``request.cpp:997-998``. They have to:
   the whole point of ``proxy.py`` is that these objects cross a zmq boundary.
   So ``__getstate__`` on the dispatch thread produces a plain tuple that
   ``depythonize`` *can* walk, under the GIL that thread already holds. That is
   the one extra step, and it is what this module charges 10.72 us for.

   Caveat, and it is a real one: ``Response.__getstate__`` calls
   ``self.getResult()``, which **throws on an errored response** -- which is
   precisely why ``base_worker.py:1267-1271`` pre-converts errors to a Python
   ``ErrorResponse`` with the comment *"tllm.Response cannot be serialized when
   it has error"*. The error path must stay in Python.

2. **On the PyTorch backend -- the one dynamo runs -- there is no nanobind
   object on the wire at all, and the payload is ALREADY a byte blob.**
   ``cpp/tensorrt_llm/nanobind/batch_manager/bindings.cpp:498-505``'s
   ``create_serialized_result`` returns ``std::make_tuple(nb::bytes(...),
   is_final)``; ``tensorrt_llm/_torch/pyexecutor/llm_request.py:890-891`` takes
   that, and ``llm_request.py:578`` ``LlmResult`` carries it as ``bytes`` inside
   the ``llm_request.py:622`` ``LlmResponse`` dataclass. The C++ side is
   deserialised **on the event loop**, at ``result.py:514-516``::

       if hasattr(response_result, "_result") and isinstance(response_result._result, bytes):
           response_result.deserialize()

   That is part of the 23.97 us, it is pure C++ work, and Rust could do it on a
   tokio thread with no GIL whatsoever. There is already a serialised
   representation; nobody has to invent a wire format for it.

3. **What ``rust-egress-full`` additionally removes.** ``ipc.py:373``
   ``pickle.dumps`` / ``ipc.py:389`` ``pickle.loads``, plus a *mandatory*
   HMAC-SHA256 over the whole buffer (``ipc.py:362-366``; ``ipc.py:51-54``
   raises if HMAC is disabled, so there is no opt-out). Replace
   ``FusedIpcQueue``'s codec with the blob from (2) plus a length-prefixed
   header and Rust can read the socket directly. The Python interpreter then
   never sees a response. :meth:`extra_report` measures the simulator's own
   ``pickle.loads`` cost per response so the size of that term is a number
   rather than a claim -- and the simulator CANNOT remove it (``fake_trtllm``
   is frozen), so ``rust-egress-full``'s measured number is understated by
   exactly that much.

What can and cannot move to Rust, per line of handler_base.py:1179-1278
-----------------------------------------------------------------------
Verdict on the response loop, line by line:

``1179  for output in res.outputs``            MOVES. ``res.outputs`` is
        ``GenerationResultBase``'s own ``self._outputs`` list, accumulated by
        ``_handle_response``; if Rust runs ``_handle_response`` it owns the list.
``1184  output_idx = getattr(output, "index", 0)``   MOVES. Integer field.
``1185  tokens_so_far = output_tokens_per_choice.get(output_idx, 0)``  MOVES.
        ``output_tokens_per_choice`` (handler_base.py:1018) is a private
        ``dict[int, int]`` cursor owned entirely by this coroutine. Nothing else
        reads it. A ``HashMap<u32, usize>`` in Rust is a strictly better home.
``1186  next_total_toks = len(output.token_ids)``    MOVES.
``1191-1194  out = {"token_ids": output.token_ids[tokens_so_far:], ...}``
        MOVES. A slice of a Vec<u32>. This is the single most expensive line in
        the stage and the most mechanical.
``1199-1205  self._extract_logprobs(output, tokens_so_far)``   MOVES. It
        delegates to ``common/backend/logprobs.py:56``
        ``extract_from_completion_output``, which slices the same cumulative
        arrays with the same cursor and reads ``.logprob`` / ``.rank`` /
        ``.decoded_token`` off each entry -- floats and ints. The one genuinely
        Python-dependent thing in that function is the ``tokenizer.decode``
        fallback (logprobs.py:117-119), and **it is unreachable from here**:
        ``handler_base.py:414-422`` calls it without a ``tokenizer``, so the
        argument defaults to ``None``. Only reached when logprobs were
        requested at all.
``1207-1210  finish_reason / stop_reason``    MOVES. Enum -> string.
``1211-1224  if self.disaggregation_mode == PREFILL: _encode_and_pack_disaggregated_params``
        **STAYS -- but is not on this path.** Guarded on PREFILL; the capture is
        a DECODE worker and the whole question is about decode. On a prefill
        worker there is exactly one response per request, so it is per-REQUEST
        in all but name.
``1226-1232  finish_reason fallback + logging.warning``   MOVES (the warning is
        a ``tracing::warn!``).
``1234-1237  num_input_tokens / total_completion_tokens``   MOVES. ``num_input_tokens``
        is ``len(request["token_ids"])``, known at submit time -- carry it once
        per request, exactly as ``base_worker.py:1426-1429`` already carries
        ``sampling_params`` to the postproc workers *"only once for each
        Request"*.
``1239-1257  prompt_tokens_details from output.request_perf_metrics``   MOVES.
        Two integer reads and a ``min()``. ``request_perf_metrics`` is in
        ``Result``'s getstate tuple (request.cpp:975), so it survives the
        conversion.
``1259-1266  out["completion_usage"]``   MOVES. Four integers.
``1277  yield out``                     BECOMES ``tx.try_send(frame)``.
``1281-1289  metrics_collector on res.finished``   Per-request in effect (fires
        once, on the final response). Can stay in Python: it does not touch the
        chunk.

Nothing in that range is Python-dependent. There is no user callback, no
tokenizer, no torch tensor, no dynamic dispatch. It is a slice, a dict literal,
two optional strings and four integers -- over state (``self._outputs``, the
per-choice cursor, ``num_input_tokens``) that Rust can own outright.

What must stay in Python -- and where it actually is
-----------------------------------------------------
All four are **per-REQUEST**, not per-response, and all four are upstream of the
response loop:

* **sampling-param translation** -- handler_base.py:1020 ``trtllm:sampling_params``
  through :1126. Builds a ``SamplingParams`` object; measured 17.42 us, once.
* **multimodal** -- handler_base.py:801-812, before submit; and the only
  per-response mention (handler_base.py:1049) is inside the
  ``sampling_params`` range, still per-request.
* **disagg codecs** -- ``_encode_and_pack_disaggregated_params``
  (handler_base.py:562) is called at :1215 under a PREFILL guard; ingress side is
  ``trtllm:setup_disagg_params`` at :975, per-request, 37.95 us.
* **logits processors** -- handler_base.py:1098-1101, per-request, and behind
  ``DYN_ENABLE_TEST_LOGITS_PROCESSOR``.

So the per-request 213 us of ingress stays exactly where it is. Only the
per-response 85.34 us moves, which is the only part that scales with tokens.

The real change, file by file
-----------------------------
1. **``lib/bindings/python/rust/push_egress.rs``** -- add the owned-frame entry
   point next to the existing one. ``ResponseSink`` (:125-139) gains
   ``fn send_owned(&self, frame: Annotated<Resp>)``, which is
   ``TypedSink::send`` (:204) with the ``decode_response`` call at :211 removed
   and everything from :213 onwards kept verbatim -- the ``sender()`` /
   ``try_send`` / ``allow_threads(blocking_send)`` sequence is already correct
   for a frame that was built elsewhere. ``ResponseSender`` (:340-367) gains a
   ``#[pymethods]`` wrapper taking an opaque ``#[pyclass] OwnedFrame`` handle,
   so the Python side can pass a frame through without the interpreter ever
   materialising the response.
2. **A new module, ``lib/bindings/python/rust/trtllm_egress.rs``.** Owns:
   ``EgressPipeline::register(client_id, prompt_tokens, sender)`` (once per
   request, from the handler), ``EgressPipeline::submit(py, state)`` (once per
   response, from ``proxy.py``'s dispatch thread -- the ``depythonize`` of
   ``Response.__getstate__``'s tuple, under the caller's GIL, mirroring
   ``decode_response`` at push_egress.rs:284), and a tokio task per shard
   holding ``RequestState { outputs: Vec<Vec<TokenId>>, cursor: HashMap<u32,
   usize>, prompt_tokens: usize }`` -- i.e. ``result.py``'s ``self._outputs``
   plus ``handler_base.py:1018``'s ``output_tokens_per_choice``. Sharding by
   ``client_id`` is required for the same reason ``base_worker.py:1437-1440``
   gives.
3. **``components/src/dynamo/trtllm/request_handlers/handler_base.py``** -- the
   ``async for res in generation_result`` block (:1158-1278) collapses to a
   registration call plus an await on completion. Ingress (:971, :975, :986,
   :1020, :1132) is untouched; so is ``_cancellation_monitor`` (:1154), which is
   per-request.
4. **``tensorrt_llm/executor/proxy.py``** -- ``dispatch_result_task``'s
   ``queue.put_nowait(res)`` (:555) and ``_SyncQueue.notify_many`` (:580) become
   ``pipeline.submit(...)`` for registered client_ids, with the existing path
   kept for everyone else. The zmq PAIR socket, the ``ManagedThread``
   (:594-602) and ``customized_gc_thresholds`` all stay.
5. **``lib/bindings/python/rust/engine.rs``** -- unchanged.
   ``invoke_generator`` (:75-120) still runs once per request and
   ``demand_driven_python_stream`` (:122-151) still advances the push
   generator once. **``python_payload.rs``** -- unchanged; it is the pull path.
6. **Only for ``rust-egress-full``:** ``tensorrt_llm/executor/ipc.py`` needs a
   codec that is not ``pickle`` (:373 dumps, :389 loads) so Rust can open the
   socket. The payload is already language-neutral on the PyTorch backend --
   see the pickle/nanobind section above -- so this is a framing change
   (length prefix + the existing HMAC at :362-366), not a schema design.

Note what is NOT needed: no ``num_postprocess_workers``, no extra processes, no
second pickle hop, and therefore nothing to revert in
``components/src/dynamo/trtllm/workers/llm_worker.py`` (``main``'s dbeaa5b166
added ``_strip_postprocess_workers`` there; it is not on this branch, and this
design does not care either way). That is the structural difference from the
``postproc-procs`` route -- which needs ``npw>0`` turned back on, dynamo's chunk
builder registered as the postproc hook, and a second pickle+HMAC round trip per
response.

Deviations, and their sign
--------------------------
1. **The simulator's ``pickle.loads`` cannot be removed.** ``fake_trtllm/ipc.py``
   is frozen, so even ``rust-egress-full`` pays a Python unpickle on the reader
   thread. Measured and reported as ``ipc_pickle_us_per_response``. Pessimistic.
2. **The real bookkeeping is Python and holds the GIL.** Building the chunk dict
   and slicing the token list happen for real (that is the point), then
   ``spin_offgil`` charges the FULL modelled cost on top rather than padding to
   it. So the off-GIL variants are charged more total work than the baseline,
   and some of their "off-GIL" stages hold the GIL for the couple of
   microseconds of real dict work. Pessimistic, twice.
3. **The conversion is ungated by the credit window** -- it has to be, or the
   reader stops draining the IPC lane and ``responses_dispatched`` freezes,
   which is the signal the benchmark uses to prove saturation (that failure was
   found by the dispatch-thread experiment). So the reader converts responses
   the loop never reaches. ``converts_per_loop_item`` reports the overhead.
   Pessimistic.
4. **``rust_egress_us`` (11.56 us) is still charged with ``spin``** in
   ``Driver._on_item`` -- core code, unchanged, and identical for every
   architecture including the baseline. On the real worker it is off-GIL, so
   every architecture here is understated by the same amount.
5. **Cores, not the GIL, may bind.** Once the GIL stops being the constraint the
   next one is CPU. With four cores pinned, 85.34 us/item of off-GIL work has a
   hard ceiling of 4e6/85.34 = 46,900 items/s no matter how good the design is.
   ``postproc-procs`` faces the same wall with its four child processes, so the
   comparison is fair, but neither number is an architecture's ceiling.

Three measurement traps, found the hard way
--------------------------------------------
Neither is a bug in the simulator's *model*; both make the standard protocol
produce numbers that are not comparable, so they are recorded here.

**A. ``spin_offgil`` is memory-bandwidth bound, so the "5 pinned runs in
parallel" protocol is invalid for off-GIL architectures.** ``costs.spin_offgil``
burns CPU by hashing a ~10 KB slice of a 1 MB buffer in a loop
(``costs.py:158-179``) -- hashlib is what releases the GIL, and that is the
whole reason it works. But it means the work is a stream of memory traffic,
whereas ``costs.spin`` is a register-only ``while _perf() < deadline: pass``.
Five concurrent pinned runs are therefore *not* independent for anything using
``spin_offgil``: they contend for LLC and memory bandwidth across the pinning
boundary. Measured, same 4 cores, same architecture:

    baseline-push        solo 9,233   in a 5-way batch 9,297   (no effect)
    rust-egress-full-bp  solo 15,019  in a 5-way batch 12,748  (-15 %)

``postproc-procs`` is unaffected for the same reason: its child processes spin
with ``spin``, not ``spin_offgil``. So a batched comparison silently handicaps
exactly the architectures this experiment is about. **Measure off-GIL
architectures serially.**

**B. ``bench.py``'s batch ladder degenerates above ~batch 600, and it only
degenerates for architectures that are fast enough to escape rung one.**
``bench._config`` (bench.py:119-134) sets ``concurrency=batch`` and
``requests=batch`` along with the engine batch, so escalating the ladder does
not only raise the offered response rate -- it also multiplies the number of
requests the loop must ADMIT, at ``prepare_request 58.46 + engine_submit
154.64 = 213 us`` each, all of it on the loop. At batch 4,000 that is 852 ms of
pure per-request loop work inside a ``DURATION_S = 12`` run, and ``_measure``'s
fixed 6 s slice lands almost entirely outside the usable region.

Reproduction: ``python3 -m egress_experiments.bench --architecture
rust-egress-full --json``. When the loop keeps up at 240/600/1500 the ladder
escalates to 4,000 and reports::

    batch                        4000
    window_s                     0.0989      <- intended 6.0
    items_in_window              1947
    work_us_per_item_on_loop     201.6       <- modelled 0.91
    arch_report.offgil_us_per_item  243.19   <- expected 85.34

Every one of those is the same artefact: the denominator (items through the
loop) collapsed while the per-request admission work did not. **Read any run
with ``batch >= 1500`` or ``window_s < 1.0`` as void**, and compare
architectures on the rung they share.

The clean way to force a shared rung is ``--cost-scale``: at ``--cost-scale 2``
every architecture here saturates at batch 240 against the same 24,000/s offer,
and ``Costs.scale`` exists for exactly this ("useful to check that conclusions
are structural rather than an artefact of one calibration", costs.py:258-260).

**C. ``spin_offgil`` under-reports the CPU it burns, worst at exactly this
granularity.** It hashes in ~39 KB blocks and checks the clock only between
them, so it always finishes the block it started; the ledger records the
REQUESTED microseconds. Measured single-threaded (see the table above
``_charge_convert``): a 10.72 us request burns 20.60 us -- **1.92x** -- and
three per-stage calls per response burn 110.97 us against a charged 85.34
(**1.30x**), while ``spin`` is accurate to 1 %. Every ``spin``-based
architecture, ``postproc-procs`` included, is therefore accounted honestly while
a naive off-GIL one silently burns 30 % more CPU than its ledger admits. This
module settles once per response per ledger, which brings it to 1.05x. Anyone
adding an off-GIL architecture should do the same, and should not compare
per-stage ``spin_offgil`` costs against ``pad_to``-based ones without correcting
for it.
"""

from __future__ import annotations

import asyncio
import functools
import os
import pickle
import queue as _queue
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from egress_experiments import architectures, loop_meter
from egress_experiments.costs import Costs, offgil_ledger, spin, spin_offgil
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import (
    Driver,
    FakeContext,
    PushDriver,
    ResponseSender,
    TokioRuntime,
    push_pump,
)
from egress_experiments.dynamo_sim.worker import SamplingParams, TrtllmWorkerHandler
from egress_experiments.fake_trtllm.aqueue import AsyncQueue, SyncQueue
from egress_experiments.fake_trtllm.engine import EngineConfig, spawn_engine
from egress_experiments.fake_trtllm.llm import FakeLLM
from egress_experiments.fake_trtllm.result import GenerationResult
from egress_experiments.nvtx_shim import range_

_perf = time.perf_counter_ns

#: Cost of one ``depythonize`` into an owned Rust value. Defaults to the
#: measured ``trtllm:push_send`` p50, which is exactly that operation on this
#: code path. See the module docstring on why this is the load-bearing choice.
CONVERT_US = float(os.environ.get("DYN_SIM_CONVERT_US", "0") or 0) or None

#: How far ahead of the loop the tokio pool may work, in responses. Bounds the
#: work done for responses the loop never reaches, so ``offgil_ledger()`` per
#: DELIVERED item stays equal to the work that was moved. ~50 ms of loop time at
#: 40,000 items/s: deep enough that loop jitter never stalls the pool.
CREDIT_WINDOW = int(os.environ.get("DYN_SIM_RUST_CREDIT", "2048"))


class _EndOfStream:
    """Producer-side terminator.

    ``_handle_response`` sets ``self._done``, and ``__anext__``
    (``result.py:1104``) tests it *before* awaiting -- so once the work runs
    ahead of the loop, that guard would make the consumer skip items still
    sitting in the queue. The pool therefore enqueues an explicit marker after
    the final chunk and the loop drains until it sees it.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<EOS>"


_EOS = _EndOfStream()


class _Credit:
    """Bounds how far the tokio pool may run ahead of the loop.

    Acquired by a pool thread once per queue entry, released on the loop once
    the entry has been consumed. Blocking here parks the thread rather than
    spinning, which is what a full ``tokio::sync::mpsc`` does
    (``push_egress.rs:249`` -- ``py.allow_threads(|| tx.blocking_send(frame))``).
    """

    __slots__ = ("_sem", "_stop")

    def __init__(self, limit: int, stop: threading.Event) -> None:
        self._sem = threading.Semaphore(limit)
        self._stop = stop

    def acquire(self) -> bool:
        """False means the run is shutting down; abandon the response."""
        while not self._stop.is_set():
            if self._sem.acquire(timeout=0.05):
                return True
        return False

    def release(self) -> None:
        self._sem.release()


class _Owned:
    """One response after the conversion: an **owned** value, no Python handle.

    This is the simulation's ``Annotated<Resp>`` -- what ``depythonize``
    produces at ``push_egress.rs:301-303``. Deliberately carries only plain
    ints/lists/strings copied out of the ``tllm.Response``: nothing here
    references the original object, so the pool threads that read it are doing
    what a tokio thread reading a Rust struct does. Contrast
    ``python_payload.rs:25``'s ``PythonPayload(Py<PyAny>)``, which keeps the
    handle and therefore needs the GIL forever after.
    """

    __slots__ = ("client_id", "new_token_ids", "is_final", "finish_reasons", "error")

    def __init__(
        self,
        client_id: int,
        new_token_ids: List[List[int]],
        is_final: bool,
        finish_reasons: Optional[List[Optional[str]]],
        error: bool = False,
    ) -> None:
        self.client_id = client_id
        self.new_token_ids = new_token_ids
        self.is_final = is_final
        self.finish_reasons = finish_reasons
        self.error = error


class _Record:
    """Per-request state the Rust side owns.

    ``tokens`` is ``GenerationResultBase._outputs``' cumulative ``token_ids``
    (result.py:513-565 accumulates into it); ``cursor`` is
    ``handler_base.py:1018``'s ``output_tokens_per_choice``; ``prompt_tokens`` is
    ``len(request["token_ids"])``, carried once per request exactly as
    ``base_worker.py:1426-1429`` already carries per-request params to the
    postproc workers.
    """

    __slots__ = ("tokens", "finish", "stop", "cursor", "prompt_tokens")

    def __init__(self, prompt_tokens: int) -> None:
        self.tokens: List[List[int]] = []
        self.finish: List[Optional[str]] = []
        self.stop: List[Optional[str]] = []
        self.cursor: Dict[int, int] = {}
        self.prompt_tokens = prompt_tokens


# ---------------------------------------------------------------------------
# The proxy side
# ---------------------------------------------------------------------------


class RustEgressLLM(FakeLLM):
    """``FakeLLM`` with the conversion moved to the front of the pipeline.

    Two kinds of thread, matching what the real change would create:

    * the **reader** -- ``proxy_dispatch_result_thread`` (``proxy.py:594-602``),
      or a Rust-owned socket thread in ``full`` mode. It unpickles the IPC
      message and converts every response into an :class:`_Owned` value. This is
      the only place a Python object is touched.
    * the **tokio pool** -- ``workers`` threads that run ``handle_response`` and
      ``build_response`` on owned values with no GIL, then ``put_nowait`` the
      finished frames and issue ONE ``notify_many`` per shard, which is
      ``proxy.py:555-580`` unchanged.

    Sharding is by ``client_id % workers`` and is sticky, for the same reason
    ``base_worker.py:1437-1440`` gives: the accumulation of cumulative
    ``token_ids`` and the per-choice cursor are per-request state and must be
    seen in order.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        costs: Costs,
        *,
        workers: int = 4,
        offgil: bool = True,
        convert_offgil: bool = False,
        convert_in_pool: bool = False,
        credit_window: int = CREDIT_WINDOW,
        reader_name: str = "proxy_dispatch_result_thread",
    ) -> None:
        super().__init__(engine_config, costs=costs)
        self.workers = max(1, workers)
        self.offgil = offgil
        self.convert_offgil = convert_offgil
        self.convert_in_pool = convert_in_pool
        self.reader_name = reader_name
        self.credit = _Credit(credit_window, self._stop) if credit_window else None
        self.credit_window = credit_window

        #: Prompt length per client_id, written under the results lock at submit
        #: time so the pool can never race the first response.
        self.prompt_lens: Dict[int, int] = {}

        self._wqueues: List["_queue.SimpleQueue"] = []
        self._wthreads: List[threading.Thread] = []
        # `+= 1` on a shared int is not atomic across threads, so one cell per
        # producer, summed at shutdown -- before harness.py reads the counters.
        self._notify_cells: List[List[int]] = [[0] for _ in range(self.workers)]
        self._frame_cells: List[List[int]] = [[0] for _ in range(self.workers)]
        self._abandoned_cells: List[List[int]] = [[0] for _ in range(self.workers)]
        self.worker_errors: List[str] = []
        #: Responses converted. Compare with items through the loop: the excess
        #: is work done for responses the loop never reached. One cell per
        #: producer (the reader is the last cell), summed at shutdown.
        self._convert_cells: List[List[int]] = [[0] for _ in range(self.workers + 1)]
        #: One real IPC batch, kept by reference so the wire-format cost can be
        #: measured after the run without perturbing it.
        self._sample_batch: Optional[List[Any]] = None

    # -- the two knobs the variants differ by ------------------------------
    #
    # Costs are ACCUMULATED per response and settled with ONE `spin` and ONE
    # `spin_offgil` call, rather than one call per stage. This is not cosmetic:
    # `spin_offgil` hashes in ~39 KB blocks and only checks the clock between
    # them (costs.py:158-179), so it always finishes the block it started and a
    # request for N us burns N + block. Measured on this box, single-threaded:
    #
    #     charged          spin_offgil actual        spin actual
    #     10.72 us         20.60 us  (1.92x)         10.90 us  (1.02x)
    #     23.97 us         30.74 us  (1.28x)         24.14 us  (1.01x)
    #     50.65 us         59.87 us  (1.18x)         50.82 us  (1.00x)
    #     three calls      110.97 us (1.30x)         85.85 us  (1.01x)
    #     one 85.34 call    89.97 us (1.05x)         85.52 us  (1.00x)
    #
    # The ledger records the REQUESTED microseconds, so the overshoot is
    # invisible to the conservation check -- three calls per response would
    # silently burn 30 % more CPU than reported, while every `spin`-based
    # architecture (including `postproc-procs`, whose child processes use
    # `pad_to`/`spin`) is accurate to 1 %. Settling once per response per
    # ledger brings the two to parity at ~5 %. Reproduce with
    # `/tmp/re/offgil_probe.py`-style timing around `costs.spin_offgil`.
    #
    # It is also the more faithful model: in Rust the three stages are
    # contiguous work on one thread, not three separately-timed regions.

    def _charge_convert(self, us: float, owed: Optional[List[float]]) -> None:
        """``depythonize`` into an owned value. GIL held, or not."""
        if owed is None:
            # Reader path: one call per response anyway, nothing to batch.
            (spin_offgil if self.convert_offgil else spin)(us)
        elif self.convert_offgil:
            owed[1] += us
        else:
            owed[0] += us

    def _charge_pool(self, us: float, owed: List[float]) -> None:
        """``handle_response`` / ``build_response`` on a tokio thread."""
        if self.offgil:
            owed[1] += us
        else:
            owed[0] += us

    @staticmethod
    def _settle(owed: List[float]) -> None:
        """Burn one response's accumulated cost: one call per ledger."""
        if owed[0]:
            spin(owed[0])
        if owed[1]:
            spin_offgil(owed[1])

    # -- lifecycle ---------------------------------------------------------

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """``FakeLLM.start`` with the pool up first and a nameable reader.

        Pool first because the reader can hand out a shard on its very first
        message. The reader is named per variant so ``work us/item by thread``
        says which thread held the GIL and which did not.
        """
        for slot in range(self.workers):
            q: "_queue.SimpleQueue" = _queue.SimpleQueue()
            self._wqueues.append(q)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(slot,),
                name=f"tokio-egress-{slot}",
                daemon=True,
            )
            self._wthreads.append(thread)
            thread.start()

        self._loop = loop or asyncio.get_event_loop()
        self.loop_thread_name = threading.current_thread().name
        self._engine = spawn_engine(self.engine_config)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name=self.reader_name, daemon=True
        )
        self._dispatch_thread.start()

    def stop_pool(self) -> None:
        """Wake anything parked on credit and join the pool. Idempotent.

        ``_wqueues`` is deliberately NOT cleared. ``on_finished`` runs while the
        reader thread is still alive -- harness.py only joins it later, in
        ``llm.shutdown()`` -- and a reader already inside ``get(timeout=0.25)``
        will finish that iteration and fan the message out. Clearing the list
        here made that an ``IndexError`` on ``self._wqueues[slot]``, which
        killed the reader thread and, when it landed between ladder rungs, lost
        the whole run's output.
        """
        self._stop.set()
        for q in self._wqueues:
            q.put(None)
        for thread in self._wthreads:
            thread.join(timeout=5.0)
        self._wthreads = []

    def shutdown(self) -> None:
        self.stop_pool()
        super().shutdown()
        self.notify_many_calls = sum(cell[0] for cell in self._notify_cells)

    @property
    def frames_built(self) -> int:
        return sum(cell[0] for cell in self._frame_cells)

    @property
    def converted(self) -> int:
        return sum(cell[0] for cell in self._convert_cells)

    @property
    def abandoned(self) -> int:
        return sum(cell[0] for cell in self._abandoned_cells)

    # -- the boundary ------------------------------------------------------

    def generate_async(
        self,
        inputs: Any = None,
        sampling_params: Any = None,
        *,
        streaming: bool = True,
        **kwargs: Any,
    ) -> GenerationResult:
        """``FakeLLM.generate_async`` plus the per-request prompt length.

        Recorded here rather than by the handler afterwards: the pool needs it
        for ``completion_usage`` and the first response can land before the
        handler executes its next bytecode. Same reason
        ``base_worker.py:1426-1429`` sends per-request params with the *first*
        Input and not later.
        """
        if self._engine is None:
            raise RuntimeError("FakeLLM.start() must be called before generate_async()")

        client_id = next(self._client_ids)
        max_tokens = getattr(sampling_params, "max_tokens", None) or (
            self.engine_config.max_tokens
        )
        n = getattr(sampling_params, "n", None) or 1

        result = GenerationResult(
            client_id,
            n=n,
            streaming=streaming,
            costs=self.costs,
            loop=self._loop,
        )
        with self._results_lock:
            self.prompt_lens[client_id] = len(inputs or [])
            self._results[client_id] = result

        with range_("trtllm:engine_submit", color="red"):
            spin(self.costs.scaled(self.costs.engine_submit_us))
            self._engine.request_link.parent.put(
                {
                    "client_id": client_id,
                    "max_tokens": int(max_tokens),
                    "submitted_ns": _perf(),
                }
            )
        self.submitted += 1
        return result

    # -- reader: unpickle, convert, fan out --------------------------------

    def dispatch_result_task(self) -> bool:
        """``proxy.py:532``, with the conversion moved here.

        What it does NOT do any more: ``put_nowait`` into the per-request
        AsyncQueue and ``notify_many``. Nothing reaches the event loop from this
        thread; the pool does that once it has a finished frame.
        """
        engine = self._engine
        if engine is None:
            return False
        # ipc.py:389 -- pickle.loads, on this thread, holding the GIL. In
        # `full` mode this is the step that would not exist; the simulator
        # cannot remove it (see extra_report's ipc_pickle_us_per_response).
        res = engine.result_link.parent.get(timeout=0.25)
        if res is None:
            return not self._stop.is_set()

        iteration = range_("_handle_responses", color="green")
        iteration.__enter__()

        batch = res if isinstance(res, list) else [res]
        if self._sample_batch is None and batch and batch[0] is not None:
            self._sample_batch = batch

        n = self.workers
        in_pool = self.convert_in_pool
        shards: List[List[Tuple[GenerationResult, Any]]] = [[] for _ in range(n)]

        for item in batch:
            if item is None:
                iteration.__exit__()
                return False  # shutdown
            self.responses_dispatched += 1
            with self._results_lock:
                result = self._results.get(item.client_id)
            if result is None:
                continue  # late response for a finalised request, proxy.py:568

            payload = item.result
            final = item.has_error() or (payload is not None and payload.is_final)
            if final:
                with self._results_lock:
                    self._results.pop(item.client_id, None)
            # `bp` variants defer the conversion into the credit window so it
            # happens exactly once per DELIVERED response; see the module
            # docstring on the ungated converter.
            shards[item.client_id % n].append(
                (result, item if in_pool else self._convert(item, n))
            )

        self.ipc_messages += 1
        self.ipc_times.append(_perf())
        self.ipc_batch_sizes.append(len(batch))

        queues = self._wqueues
        for slot, shard in enumerate(shards):
            if shard and slot < len(queues):
                queues[slot].put(shard)

        iteration.__exit__()
        return True

    # -- the conversion point ----------------------------------------------

    def _convert(
        self, item: Any, slot: int, owed: Optional[List[float]] = None
    ) -> _Owned:
        """``push_egress.rs:284`` ``decode_response``, moved to the front.

        The one and only place a Python object is read. Everything downstream
        sees :class:`_Owned`, which aliases nothing the interpreter owns -- the
        token lists are copied, not referenced. That is the difference between
        ``Annotated<Resp>`` (push_egress.rs:301-303) and
        ``PythonPayload(Py<PyAny>)`` (python_payload.rs:25), and it is why the
        tokio side never needs the GIL again.

        On the real worker this runs under the GIL the caller already holds:
        the dispatch thread's, straight after ``ipc.py:389``'s ``pickle.loads``.
        """
        with range_("pybridge.convert_response", color="magenta"):
            payload = item.result
            if item.has_error() or payload is None:
                owned = _Owned(item.client_id, [], True, None, error=True)
            else:
                owned = _Owned(
                    item.client_id,
                    [list(toks) for toks in payload.new_token_ids],
                    bool(payload.is_final),
                    list(payload.finish_reasons) if payload.finish_reasons else None,
                )
            self._charge_convert(
                self.costs.scaled(CONVERT_US or self.costs.push_send_us), owed
            )
            self._convert_cells[slot][0] += 1
        return owned

    # -- the tokio pool: no Python object is reachable from here -----------

    def _worker_loop(self, slot: int) -> None:
        q = self._wqueues[slot]
        records: Dict[int, _Record] = {}
        while True:
            shard = q.get()
            if shard is None:
                return
            try:
                if not self._run_shard(records, shard, slot):
                    return
            except Exception as exc:  # pragma: no cover - defensive
                self.worker_errors.append(f"{type(exc).__name__}: {exc}")
                return

    def _run_shard(
        self,
        records: Dict[int, _Record],
        shard: List[Tuple[GenerationResult, Any]],
        slot: int,
    ) -> bool:
        """Both moved stages for one shard, then ONE notify for the shard.

        The notify comes after the whole shard, exactly as ``proxy.py:580``'s
        comes after the whole message: N responses still share one event-loop
        ready-deque entry. There are up to ``workers`` of them per IPC message
        instead of one, which the benchmark's ``deque entries/item`` reports.
        """
        async_queues: List[SyncQueue] = []
        event_loop: Optional[asyncio.AbstractEventLoop] = None
        credit = self.credit

        for result, owned in shard:
            # One credit per queue entry, taken BEFORE the work: the point is
            # not to do the work at all, not to do it and then wait.
            if credit is not None and not credit.acquire():
                self._abandoned_cells[slot][0] += 1
                break

            # [gil_us, offgil_us] owed for this response, settled once below.
            owed = [0.0, 0.0]
            if self.convert_in_pool:
                # Inside the credit window, so exactly one conversion per
                # DELIVERED response -- what a backpressured channel gives.
                owned = self._convert(owned, slot, owed)

            record = records.get(owned.client_id)
            if record is None:
                record = _Record(self.prompt_lens.get(owned.client_id, 0))
                records[owned.client_id] = record

            frames = self._build(record, owned, owed)
            self._settle(owed)
            if credit is not None:
                for _ in range(len(frames) - 1):
                    if not credit.acquire():
                        break

            queue = result.queue
            for frame in frames:
                queue.put_nowait(frame)  # deque append -- the loop is untouched
            self._frame_cells[slot][0] += len(frames)
            if owned.is_final:
                queue.put_nowait(_EOS)
                records.pop(owned.client_id, None)
            async_queues.append(queue)
            event_loop = event_loop or queue.loop

        if async_queues:
            try:
                SyncQueue.notify_many(event_loop, async_queues)
            except AsyncQueue.EventLoopShutdownError:
                return False
            self._notify_cells[slot][0] += 1
        return True

    def _build(
        self, record: _Record, owned: _Owned, owed: List[float]
    ) -> List[Dict[str, Any]]:
        """``handle_response`` + ``build_response``, on owned data.

        Line-for-line the same bookkeeping as ``fake_trtllm/result.py``'s
        ``_handle_response_impl`` and ``dynamo_sim/worker.py``'s
        ``trtllm:build_response`` range -- the work is moved, not reduced. The
        difference is that it reads :class:`_Owned` and :class:`_Record`, both
        of which are plain data this thread owns, so on the real worker none of
        it needs the GIL.

        The modelled cost is ACCUMULATED into ``owed`` and burned once by
        :meth:`_settle`, for the block-quantisation reason given above
        ``_charge_convert``. It is charged in FULL on top of the real
        bookkeeping rather than padded down to it (``pad_to`` charges the GIL
        ledger, which is the wrong ledger here), so this over-charges by the
        couple of microseconds of real dict work -- in the safe direction.
        """
        costs = self.costs
        if owned.error:
            with range_("rust:handle_response", color="red"):
                self._charge_pool(costs.scaled(costs.handle_response_us), owed)
            with range_("rust:build_response", color="yellow"):
                out = {"token_ids": [], "index": 0, "finish_reason": "error"}
                self._charge_pool(costs.scaled(costs.build_response_us), owed)
            return [out]

        # ---- handle_response (result.py:454, reached from :1043) ----------
        with range_("rust:handle_response", color="red"):
            for idx, new_tokens in enumerate(owned.new_token_ids):
                while idx >= len(record.tokens):
                    record.tokens.append([])
                    record.finish.append(None)
                    record.stop.append(None)
                # Cumulative, exactly as TRT-LLM accumulates into
                # CompletionOutput (result.py:552-565).
                record.tokens[idx].extend(new_tokens)
                if owned.finish_reasons and idx < len(owned.finish_reasons):
                    if owned.finish_reasons[idx]:
                        record.finish[idx] = owned.finish_reasons[idx]
            self._charge_pool(costs.scaled(costs.handle_response_us), owed)

        # ---- trtllm:build_response (handler_base.py:1183-1266) ------------
        frames: List[Dict[str, Any]] = []
        for output_idx in range(len(record.tokens)):
            with range_("rust:build_response", color="yellow"):
                tokens = record.tokens[output_idx]
                tokens_so_far = record.cursor.get(output_idx, 0)
                next_total_toks = len(tokens)

                out: Dict[str, Any] = {
                    "token_ids": tokens[tokens_so_far:next_total_toks],
                    "index": output_idx,
                }
                finish_reason = record.finish[output_idx]
                if finish_reason:
                    out["finish_reason"] = finish_reason
                if record.stop[output_idx]:
                    out["stop_reason"] = record.stop[output_idx]

                if out.get("finish_reason") or owned.is_final:
                    if not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                    total_completion_tokens = sum(len(t) for t in record.tokens)
                    out["completion_usage"] = {
                        "prompt_tokens": int(record.prompt_tokens),
                        "completion_tokens": int(total_completion_tokens),
                        "total_tokens": int(
                            record.prompt_tokens + total_completion_tokens
                        ),
                        "prompt_tokens_details": None,
                    }

                self._charge_pool(costs.scaled(costs.build_response_us), owed)

            record.cursor[output_idx] = next_total_toks
            frames.append(out)
        return frames


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------


class RustEgressHandler(TrtllmWorkerHandler):
    """The response loop when the frame is already an owned Rust value.

    Everything up to and including ``generate_async`` is byte-for-byte the
    baseline's -- this architecture changes the egress half only. What is gone
    from the loop is the whole ``for output in res.outputs: with
    range_("trtllm:build_response")`` block **and** the ``handle_response`` that
    ``GenerationResult.__anext__`` used to trigger via ``_aresult_step``.

    What is left is one queue pop and one ``yield``, which the real
    ``push_egress_capable`` decorator turns into ``response_sender.send(frame)``
    (``push_egress.py:187-194``). That send has nothing to depythonize, so it is
    a handle move -- see :class:`OwnedSender`.
    """

    async def _generate_locally_impl(
        self, request: dict, context: Any
    ) -> AsyncGenerator[dict, None]:
        record = self.records.get(request["id"])
        if record is not None and not record.admitted_ns:
            record.admitted_ns = _perf()

        # Ingress unchanged, same NVTX names: all four stages are per-REQUEST
        # and none of them moves. See the module docstring's "what must stay".
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
        generation_result = self.llm.generate_async(
            inputs=request.get("token_ids"),
            sampling_params=sampling_params,
            disaggregated_params=None,
            streaming=True,
            trace_headers=None,
            scheduling_params=None,
            priority=0.5,
            cache_salt=None,
        )

        # Drains the AsyncQueue directly rather than `async for res in
        # generation_result`: same await, same single ready-deque entry per
        # notify_many, but _handle_response is not called here because the pool
        # already ran it off-GIL.
        aqueue = generation_result.aqueue
        credit = getattr(self.llm, "credit", None)

        while True:
            item = await aqueue.get()
            if item is _EOS:
                return
            if credit is not None:
                # Hand the pool back its room to work, on the loop, as soon as
                # the entry is off the queue.
                credit.release()
            self.responses_yielded += 1
            yield item


# ---------------------------------------------------------------------------
# The Rust side
# ---------------------------------------------------------------------------


class OwnedSender(ResponseSender):
    """``TypedSink::send`` when the frame is already ``Annotated<Resp>``.

    ``push_egress.rs:204-254`` is two things: ``decode_response`` (:211, the
    ``depythonize``) and ``tx.try_send(frame)`` (:221). Only the first costs
    anything, and in this architecture it has already happened -- on the
    dispatch thread, at the front of the pipeline. What is left is the enqueue,
    which the shipped code already documents as free: *"`try_send` never
    blocks, so there is no reason to drop the GIL for it, and
    dropping/reacquiring it would cost more than the send"* (push_egress.rs:219-220).

    So this overrides ``send`` to tick the meter and hand the frame over
    **without** the ``push_send_us`` charge. That is not deleted work: the
    identical 10.72 us is spun on the reader thread in
    :meth:`RustEgressLLM.dispatch_result_task`, which is the whole point of the
    experiment. The real change is a ``send_owned(&OwnedFrame)`` method on
    ``ResponseSender`` taking a ``#[pyclass]`` handle, so the response itself
    never becomes a Python object -- only an opaque capsule crosses.
    """

    def send(self, obj: Any) -> None:
        # The last point the loop touches the item, so this is where the
        # benchmark's tick goes -- same rule as the baseline's send.
        loop_meter.item()
        self.sends += 1
        self.send_threads.append(threading.current_thread().name)
        self._deliver(obj)


class RustEgressDriver(PushDriver):
    """``PythonPushEngine`` with an :class:`OwnedSender`.

    Identical to ``PushDriver.run`` in every other respect: one
    ``spawn_blocking`` GIL acquisition per REQUEST for ``invoke_generator``
    (engine.rs:85-115), ONE ``run_coroutine_threadsafe`` for the whole request,
    and a tokio-side consumer that never touches the loop.
    """

    async def run(self, request: dict, record: RequestRecord) -> None:
        context = FakeContext(request["id"])
        record.accepted_ns = _perf()

        sink: asyncio.Queue = asyncio.Queue()
        sender = OwnedSender(self.tokio.loop, sink, self.costs)
        self.senders.append(sender)
        context.response_sender = sender

        stream = await self.spawn_blocking(
            functools.partial(
                self.handler.generate, request, context, response_sender=sender
            )
        )
        anext = stream.__anext__

        counter = [0]
        self.loop_handoffs += 1
        pump = asyncio.run_coroutine_threadsafe(push_pump(anext, counter), self.py_loop)
        consumer = asyncio.ensure_future(self._consume(sink, record))

        try:
            try:
                await asyncio.wrap_future(pump)
            except Exception as exc:  # pragma: no cover - defensive
                self.errors.append(f"{type(exc).__name__}: {exc}")
                sender.close()
            self.fallback_yields += counter[0]
            await consumer
        except BaseException:
            # Includes CancelledError, which is how --max-backlog stops an
            # overloaded run.
            for pending in (pump, consumer):
                try:
                    pending.cancel()
                except RuntimeError:
                    pass
            raise


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class _RustEgress(architectures.Architecture):
    """Base for every point on the curve."""

    egress = "push"
    workers = 4
    #: handle_response + build_response off the GIL (Rust) or on it (control).
    offgil = True
    #: The depythonize off the GIL too -- i.e. Rust owns the IPC socket.
    convert_offgil = False
    #: Convert inside the credit window (one per delivered response) instead of
    #: on the reader (one per response the engine emitted).
    convert_in_pool = False
    reader_name = "proxy_dispatch_result_thread"

    def __init__(self) -> None:
        self._llm: Optional[RustEgressLLM] = None

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        self._llm = RustEgressLLM(
            engine_config,
            costs,
            workers=self.workers,
            offgil=self.offgil,
            convert_offgil=self.convert_offgil,
            convert_in_pool=self.convert_in_pool,
            credit_window=CREDIT_WINDOW,
            reader_name=self.reader_name,
        )
        return self._llm

    def build_handler(
        self,
        llm: FakeLLM,
        costs: Costs,
        records: Dict[str, RequestRecord],
    ) -> Any:
        return RustEgressHandler(llm, costs=costs, records=records)

    def build_driver(
        self, handler: Any, py_loop: Any, tokio: TokioRuntime, costs: Costs
    ) -> Driver:
        return RustEgressDriver(handler, py_loop, tokio, costs)

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        # Rule 5: everything this architecture spawned is stopped here. Also
        # releases any pool thread parked on the credit semaphore.
        if isinstance(llm, RustEgressLLM):
            llm.stop_pool()

    # -- reporting ---------------------------------------------------------

    def _pickle_us_per_response(self) -> Optional[float]:
        """What the wire format costs, measured after the run.

        ``rust-egress-full``'s whole premise is that Rust reads the socket, so
        this ``pickle.loads`` disappears. ``fake_trtllm/ipc.py`` is frozen and
        the simulator cannot remove it, so measure it instead: this is the
        amount by which ``rust-egress-full``'s number is understated.

        Timed here, off the measurement window, against a real batch captured by
        reference during the run. Indicative only: the simulator's ``Response``
        is a dataclass, while the real one is a nanobind object whose
        ``__setstate__`` (``request.cpp:1011-1019``) reconstructs C++ state, and
        the real path also verifies a mandatory HMAC-SHA256 over the whole
        buffer (``ipc.py:362-366``, ``ipc.py:51-54``).
        """
        llm = self._llm
        if llm is None or not llm._sample_batch:
            return None
        batch = llm._sample_batch
        try:
            blob = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # pragma: no cover - defensive
            return None
        reps = 20
        start = _perf()
        for _ in range(reps):
            pickle.loads(blob)
        elapsed_us = (_perf() - start) / 1000.0
        return elapsed_us / reps / max(1, len(batch))

    def extra_report(self) -> Dict[str, Any]:
        llm = self._llm
        if llm is None:
            return {}
        items = max(1, loop_meter.count())
        offgil = offgil_ledger()
        offgil_total = sum(offgil.values())
        costs = llm.costs
        moved = costs.scaled(costs.handle_response_us + costs.build_response_us)
        if llm.convert_offgil:
            moved += costs.scaled(CONVERT_US or costs.push_send_us)

        report: Dict[str, Any] = {
            "tokio_threads": llm.workers,
            "convert_at": (
                (
                    "tokio pool, backpressured"
                    if llm.convert_in_pool
                    else "rust ipc reader"
                )
                + (" (NO GIL)" if llm.convert_offgil else " (GIL already held)")
            ),
            "moved_work": "spin_offgil (Rust)"
            if llm.offgil
            else "spin (GIL) -- CONTROL",
            "convert_us": round(costs.scaled(CONVERT_US or costs.push_send_us), 2),
            # Conservation: this is where the 74.62 us/item went.
            "offgil_us_per_item": round(offgil_total / items, 2),
            "offgil_us_per_item_expected": round(moved, 2),
            "offgil_us_total": round(offgil_total),
            "offgil_by_thread_us_per_item": {
                k: round(v / items, 2) for k, v in offgil.items()
            },
            "converts_per_loop_item": round(llm.converted / items, 3),
            "frames_built": llm.frames_built,
            "credit_window": llm.credit_window or "unbounded",
        }
        pickle_us = self._pickle_us_per_response()
        if pickle_us is not None:
            report["ipc_pickle_us_per_response"] = round(pickle_us, 2)
        if llm.abandoned:
            report["abandoned_at_shutdown"] = llm.abandoned
        if llm.worker_errors:
            report["worker_errors"] = llm.worker_errors[:2]
        return report


class RustEgress(_RustEgress):
    name = "rust-egress"
    description = "depythonize on the dispatch thread; handle+build off-GIL on 4 tokio"
    workers = 4


class RustEgressW1(_RustEgress):
    name = "rust-egress-w1"
    description = "same, 1 tokio thread -- shows the off-GIL pool is the constraint"
    workers = 1


class RustEgressW8(_RustEgress):
    name = "rust-egress-w8"
    description = "same, 8 tokio threads"
    workers = 8


class RustEgressFull(_RustEgress):
    name = "rust-egress-full"
    description = "Rust owns the IPC socket: no pickle, conversion also off-GIL (4)"
    convert_offgil = True
    reader_name = "rust_ipc_reader"
    workers = 4


class RustEgressFullW8(_RustEgress):
    name = "rust-egress-full-w8"
    description = "Rust owns the IPC socket, 8 tokio threads"
    convert_offgil = True
    reader_name = "rust_ipc_reader"
    workers = 8


class RustEgressGil(_RustEgress):
    name = "rust-egress-gil"
    description = "ABLATION: identical topology, moved work HOLDS the GIL (4 threads)"
    offgil = False
    workers = 4


class RustEgressBp(_RustEgress):
    name = "rust-egress-bp"
    description = "rust-egress with one conversion per DELIVERED response (4 tokio)"
    convert_in_pool = True
    workers = 4


class RustEgressFullBp(_RustEgress):
    name = "rust-egress-full-bp"
    description = "rust-egress-full with one conversion per DELIVERED response (4)"
    convert_offgil = True
    convert_in_pool = True
    reader_name = "rust_ipc_reader"
    workers = 4


class RustEgressGilBp(_RustEgress):
    name = "rust-egress-gil-bp"
    description = "ABLATION with backpressured convert: moved work HOLDS the GIL (4)"
    offgil = False
    convert_in_pool = True
    workers = 4


for _factory in (
    RustEgress,
    RustEgressW1,
    RustEgressW8,
    RustEgressFull,
    RustEgressFullW8,
    RustEgressGil,
    RustEgressBp,
    RustEgressFullBp,
    RustEgressGilBp,
):
    architectures.register(_factory)

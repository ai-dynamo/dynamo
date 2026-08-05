# Simulated asyncio / GIL path — `egress_experiments`

The companion to `ASYNCIO_GIL_PATH.md`. That diagram contrasts **trtllm-serve vs
dynamo** on real hardware; this one draws what the simulation actually builds,
and contrasts **pull vs push** — the A/B the simulation exists to run.

Ingress and engine are drawn once, full width, because they are *identical*
between the two egress paths by construction. Only the last hop differs.

```
════════════════════════════════════════════════════════════════════════════════════════════════
                     SIMULATED ASYNCIO / GIL PATH — egress_experiments
        pull vs push · identical ingress, engine and stage costs · npw=0 · 3 processes
════════════════════════════════════════════════════════════════════════════════════════════════

 INGRESS ════════════════════════════════════════════════════════════════════════════════════════

     harness.orchestrate()          arrival ∈ {constant, poisson, closed}
     "requests off the wire"        qps default = batch / (max_tokens × iteration_s)
              │                                                          harness.py
              ▼
   ┌────────────────────────────────────────────┐
   │ TOKIO STAND-IN  (a 2nd loop, own thread)   │   ⚠ Python, so it HOLDS the GIL where real
   │ PullDriver / PushDriver                    │     tokio does not — charges push only.
   │                          rust_bridge.py    │     ⚠ ONE worker, not 8: the count of
   └───────────────────┬────────────────────────┘       contenders is understated.
                       │
   ┌───────────────────▼────────────────────────┐   engine.rs:85-114 — BOTH paths, per REQUEST.
   │ spawn_blocking POOL  (real threads)        │   tokio::task::spawn_blocking + with_gil:
   │   pybridge.invoke_generator                │   build the request object, call generate().
   │   → GIL taken OFF the loop thread          │   The generator body has not run yet.
   └───────────────────┬────────────────────────┘   Diagram: "GIL ACQUISITION (pyo3,
                       │                             spawn_blocking pool)  p50 1.05 ms"
                       │  asyncio.run_coroutine_threadsafe(...)
                       │  → call_soon_threadsafe → ONE ready-deque entry
   ╔═══════════════════▼════════════════════════════════════════════════════════════════════════╗
   ║  THE ONE ASYNCIO DEQUE — shared by EVERY request AND every response notification           ║
   ║                                                                                            ║
   ║  ▶ a request sits here, already off the wire, waiting for the loop to drain to it          ║
   ║    before it can reach the engine                                                          ║
   ║                                                                                            ║
   ║      entries per response:   PULL 1.062          PUSH 0.062        ◀── the whole argument  ║
   ║      admission wait p90:     PULL 4.86 ms        PUSH 1.32 ms                              ║
   ║      with 42 GIL contenders: PULL 66.09 ms       PUSH 18.76 ms                             ║
   ╚═══════════════════╤════════════════════════════════════════════════════════════════════════╝
                       │
   ┌───────────────────▼────────────────────────┐
   │ HANDLER, ON THE LOOP        worker.py      │   trtllm:normalize_request      1.16 µs
   │ TrtllmWorkerHandler.generate_locally       │   trtllm:setup_disagg_params   37.95 µs
   │ (wrapped by the REAL push_egress_capable)  │   trtllm:prepare_input          1.93 µs
   └───────────────────┬────────────────────────┘   trtllm:sampling_params       17.42 µs
                       │                            ─────────────────────────────────────
                       ▼                                                        58.46 µs
              llm.generate_async(...)   ◀── THE MOCKED BOUNDARY.  fake_trtllm/llm.py
                       │                    Accepts handler_base's full signature, returns a
                       ▼                    GenerationResult handle. The engine eats it.
              trtllm:engine_submit  154.64 µs
                       │
                       │  IPC · zmq PAIR (or socketpair) — PAIR because npw=0
 ════════════════════════════════════════════════════════════════════════════════════════════════
 ENGINE (opaque, its OWN PROCESS — its GIL is not the app's)              fake_trtllm/engine.py
 ════════════════════════════════════════════════════════════════════════════════════════════════
   ║  batch:      --batch-total N   |   --batch-per-rank N --ranks R  (attention-DP geometry)   ║
   ║  iteration:  ConstantIteration(52.1 ms)  |  BatchDependentIteration(22.296 + 0.0298·batch) ║
   ║                                                                                            ║
   ║  per iteration: EVERY in-flight request generates one token                                ║
   ║                                                                                            ║
   ║  ┌── stream_interval ─────────────────────────────────────────────────────────────────┐   ║
   ║  │ a token becomes a RESPONSE only every S iterations, carrying the S tokens since    │   ║
   ║  │ the last one.  TOKEN rate unchanged; RESPONSE rate divided by S.                   │   ║
   ║  │                                                                                     │  ║
   ║  │      responses/iteration = batch / S                                                │  ║
   ║  │                                                                                     │  ║
   ║  │      si=40  →  7888/40 =  197.2/iter →   3,815/s →   32.6 % of loop capacity  ✓    │  ║
   ║  │      si=1   →  7888/ 1 = 7888.0/iter → 152,603/s → 1302.3 % ── CANNOT KEEP UP      │  ║
   ║  └─────────────────────────────────────────────────────────────────────────────────────┘  ║
   ║                                                                                            ║
   ║  ship the whole iteration as ONE IPC message                                               ║
   ║                              ← handle_for_ipc_batched (base_worker.py:1252)                ║
 ════════════════════════╤═══════════════════════════════════════════════════════════════════════
                         │  one message = 197 responses  (batch 7888 / si 40)
 EGRESS ═════════════════▼═══════════════════════════════════════════════════════════════════════
                         │
   ┌─────────────────────▼──────────────────────┐
   │ proxy_dispatch_result_thread (NOT the loop)│  port of proxy.py:532
   │   put_nowait × 197   → deque appends, FREE │  the loop never sees them
   │   notify_many × 1    → ONE deque entry     │  ← 197 responses share one entry
   │                                            │
   │   BACKLOG = put_nowait'd − delivered       │  si=40: flat, +1/s
   │   ▶ this IS the asyncio queue, measured    │  si=1:  +17,963/s, unbounded
   └─────────────────────┬──────────────────────┘
                         │
   ┌─────────────────────▼──────────────────────────────────────────────────────────────────────┐
   │ ON THE ASYNCIO LOOP — __anext__ → _aresult_step → aqueue.get() → _handle_response()        │
   │   handle_response         23.97 µs      ← runs HERE, on the loop, not on the dispatch thread│
   │   trtllm:build_response   50.65 µs      ← inline, because npw = 0                          │
   └────────────────────────────────┬─────────────────────────┬─────────────────────────────────┘
                                    │                         │
        ┌───────────────────────────▼──────────┐  ┌───────────▼─────────────────────────────────┐
        │             PULL                     │  │              PUSH                           │
        │  demand_driven_python_stream         │  │  PythonPushEngine + ResponseSender          │
        │  ──────────────────────────          │  │  ─────────────────────────────────          │
        │  yield out                           │  │  sender.send(out)      +10.72 µs            │
        │    → Rust takes the GIL, calls       │  │    → converts under the GIL we ALREADY      │
        │      __anext__, schedules onto the   │  │      hold, hands to tokio out of band       │
        │      loop, parks, is woken, takes    │  │    → NO deque entry, NO tokio thread ever   │
        │      the GIL again to depythonize    │  │      touches Python                         │
        │                                      │  │                                             │
        │  pybridge.anext_call    (tokio thr)  │  │  no tokio thread ever touches Python        │
        │  pybridge.decode_response (blocking) │  │                                             │
        │  2 off-loop GIL acq PER RESPONSE     │  │  1 GIL acquisition, an EXISTING one         │
        │  1 deque entry      PER RESPONSE     │  │  1 deque entry    PER REQUEST               │
        │  measured: 1.062 spawn_blocking/resp │  │  measured: 0.062 spawn_blocking/resp        │
        │                                      │  │                                             │
        │  loop cost   74.62 µs / response     │  │  loop cost   85.34 µs / response            │
        │  ═══ 2 STAGES ═══                    │  │  ═══ 3 STAGES ═══            = 44.0× serve  │
        └───────────────────┬──────────────────┘  └───────────────────┬─────────────────────────┘
                            │                                         │
                            ▼                                         ▼
              Rust egress, ON the GIL here          Rust egress, off the loop (tokio)
              chunk+encode+publish 11.56 µs         chunk+encode+publish 11.56 µs
                            │                                         │
                            ▼                                         ▼
                        "client"                                  "client"
════════════════════════════════════════════════════════════════════════════════════════════════
```

## The `stream_interval` knob

Everything in the ENGINE box below `batch` exists because of one line in
`server-gen-si40.yaml`:

```
stream_interval: 40
```

It is what makes the difference between a loop at 32 % and a loop that cannot
keep up at all. TRT-LLM's own docs say to raise it *"when the resource
bottleneck is on the CPU side"* — this diagram is a picture of that bottleneck.

At si=1 the achieved QPS collapses from 382 to 28 and the backlog climbs
without bound; the run has to be stopped with `--max-backlog`. Nothing else
changes: same batch, same iteration time, same token rate, same costs.

## Reading the two columns

Push costs **more** loop microseconds per response (85.34 vs 74.62) and that is
not a defect — it matches the capture, whose post-push-egress figure is 85.34.
The win is not in loop cost. It is in the two rows in the deque box: pull adds a
ready-deque entry and two cross-thread GIL acquisitions *per response*, push adds
them *per request*. At 89 tokens per request that is an 89× reduction in
cross-thread hand-offs, and it only converts into latency when there is
contention to be avoided — hence the third row, which needs `--gil-noise 42`.

## Where the numbers come from

Stage costs are p50s measured on `nsys_355778_disagg_gen-rank0.sqlite`; the
deque-box figures are from simulation runs. Both are re-derivable:

```bash
python3 -m egress_experiments.capture_params <capture.sqlite>   # stage costs
python3 -m egress_experiments.run_experiment --egress both      # deque figures
```

## What is drawn but not real

- **The engine** is a child process that sleeps for an iteration and emits one
  token per in-flight request. No GPU, no scheduler, no KV.
- **The tokio stand-in** is Python and holds the GIL where real tokio does not.
  It charges the push path for work that is free on the real worker, so every
  push win drawn here is a lower bound.
- **`Holding GIL` / `Waiting for GIL`**, which the original diagram's aggregates
  table rests on, do not appear at all — they need a real 50-thread CPython
  process under nsys.

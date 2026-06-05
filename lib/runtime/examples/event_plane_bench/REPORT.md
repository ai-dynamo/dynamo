# DIS-2172 — NATS vs ZMQ Event Plane: latency & brokerless scaling

**Status:** in progress (main sweep complete; M-sweep running).
**Owner:** Zhongdao Ren. **Sibling:** DIS-1964. **North star:** DYN-2941 (default the event plane to ZMQ).
**Date:** 2026-06-01.

---

## TL;DR (measured on `main` @ 9f36ec2ab1b, single host)

- On a like-for-like comparison (**NATS Core event-plane vs ZMQ event-plane**, identical
  msgpack `EventEnvelope`, only `DYN_EVENT_PLANE` differs), **ZMQ shows no latency penalty**
  up to **128 publishers** with one subscriber:
  - **FPM (forward-pass-metrics):** ZMQ is consistently **~25% faster** (p50 ~30µs vs NATS ~40µs).
  - **KV events:** NATS is slightly faster at large N (p50 ~45µs vs ZMQ ~55µs) — same order of magnitude, ~10µs.
- **Zero dropped events / zero sequence gaps** for both transports across the entire N=1..128 range.
  ZMQ's fail-fast `SNDTIMEOUT=0` did **not** drop at this scale/rate.
- **Brokerless cost surfaces on the subscriber (M) axis, as latency — not as drops.** Publisher axis
  scales fine (N→128, M=1: flat latency, zero drops). But with N=32 and router replicas M=1→8, ZMQ's
  connections explode (361→1284, the N×M mesh) and its p50 crosses above NATS: KV-events ~1.6× NATS at
  M=8 (133 vs 84µs), FPM crosses ≈M=4. Still **zero drops** throughout — graceful, not cliff-edge.
- **Bottom line so far:** at single-host scale, **latency is not a blocker** for making ZMQ the default.
  The open risks are (a) the N×M connection fan-out with many router replicas, (b) real multi-host
  NIC/CPU saturation, (c) large payloads vs the NATS 1 MB cap — see §6.

> Every number below is **measured** unless tagged *(est.)*. Each is reproducible from §3.6.

---

## 1. Background & question

Dynamo's **event plane** is a pub/sub broadcast bus carrying: KV-router events, worker load
metrics, **Forward-Pass-Metrics (FPM)** for the Planner, and inter-router replica syncs. It has two
transports: **NATS** (broker) and **ZMQ** (brokerless). DYN-2941 wants to drop the hard NATS
dependency and default to ZMQ. DIS-2172 must produce the evidence: *does ZMQ carry a latency/overhead
penalty, and how does it scale with #workers / #pub-sub connections (brokerless is the open question)?*

## 2. Key clarifications (verified in source)

1. **Two independent "ZMQ" paths — do not confuse them.**
   - *(a) engine → worker KV ingest* (`KvEventSourceConfig::Zmq`, `zmq_listener`): vLLM/SGLang push raw
     KV events to the worker over ZMQ. **Almost always ZMQ**, unrelated to `DYN_EVENT_PLANE`. (This is
     what `kv_event_sniffer.py` taps.)
   - *(b) worker → router/planner broadcast* — **the event plane**, selected by `DYN_EVENT_PLANE=nats|zmq`.
     **This is what DIS-2172 measures.** The mocker-based harness drives (b) directly and bypasses (a).
2. **Default transport depends on the discovery backend** (`distributed.rs:resolve_event_transport_kind`):
   unset `DYN_EVENT_PLANE` → `file`/`mem` backend = **ZMQ**, `etcd`/`kubernetes` = **NATS**. Default
   backend is `etcd`, so real/distributed deployments still default to **NATS** today — which is exactly
   why DYN-2941 (flip default to ZMQ) is still Todo and needs this evidence.
3. **Clean comparison = NATS Core event-plane vs ZMQ event-plane**, both using the identical msgpack
   `EventEnvelope`. We deliberately do **not** compare against the JetStream KV path (serde_json,
   ~40–50% larger payloads) — that would conflate serialization with transport.

## 3. Methodology

### 3.1 Harness
```
frontend (round-robin router)
  ├── 1 mocker process, --num-workers N   (N independent event-plane publishers, one tokio runtime)
  ├── loadgen.py  (asyncio httpx, sustained concurrent chat requests → workers keep doing forward passes)
  └── M × event_plane_bench_sub           (Rust subscribers; measure publish→deliver latency + seq gaps)
```
The **mocker** (`python -m dynamo.mocker`) is a pure simulator — no GPU/engine — that publishes real
event-plane traffic (KV events + FPM). Request load is required: idle mockers emit only a 1 Hz FPM
heartbeat, so `loadgen.py` keeps them busy to generate representative rates.

### 3.2 Instrumentation (the only change to dynamo)
- Added `published_at_ns: u64` to `EventEnvelope` (`traits.rs`), stamped at `publish_bytes`
  (`mod.rs`) with `current_timestamp_ns()`. `#[serde(default)]` keeps the msgpack wire format
  backward-compatible. This is the **only** change to dynamo's runtime.
- Measurement lives in a separate bin `event_plane_bench_sub` (a `lib/runtime/examples` member) that
  subscribes via the public `EventSubscriber` API, computes `recv_ns − published_at_ns`, and reports
  full-sample percentiles (p50/p90/p95/p99) plus per-publisher sequence-gap counts. We deliberately
  did **not** wire into dynamo's Prometheus registry: smaller blast radius, exact percentiles.

### 3.3 Latency definition
One-way **publish→deliver** latency = subscriber receive time − publisher `published_at_ns`. **Same host
only**: publisher and subscriber share `CLOCK_REALTIME`, so the delta is valid (no clock-skew term).
Cross-host would need PTP and is out of scope here (see §6).

### 3.4 Fairness controls
- Everything identical across transports **except** `DYN_EVENT_PLANE`.
- ZMQ runs **remove `NATS_SERVER`** and set `DYN_EVENT_PLANE=zmq` so the run is genuinely NATS-free
  (DGH-900: otherwise a NATS connection is still opened).
- Throughput/loss measured via **sequence gaps** (valid on both planes), **not**
  `dynamo_component_kv_publisher_zmq_events_total` (ZMQ-only — DGH-889/DYN-3066).
- Both transports use the same msgpack `EventEnvelope` and the same codec.

### 3.5 Environment
- dynamo `main` @ `9f36ec2ab1b`, **release** build, instrumented bindings in a throwaway venv.
- Local **nats 2.11.4** (:4222) + **etcd 3.6.1** (:2379), reused. Single host, CPython 3.10.
- Driver model: `Qwen/Qwen2.5-0.5B` (tokenizer only); mocker `--speedup-ratio 10`; loadgen concurrency 32.
- *(est.)* host = the dev box; exact CPU/NIC not pinned — single-host loopback, so NIC is not exercised
  (a known gap vs the multi-host question, §6).

### 3.6 Reproduce
```bash
# infra: reuse local nats(4222)+etcd(2379), or:  docker compose -f dynamo/dev/docker-compose.yml up -d
# build instrumented bindings into a throwaway venv (no pollution of the main env):
uv venv /tmp/dis2172-venv && source /tmp/dis2172-venv/bin/activate
maturin develop --uv --release -m dynamo/lib/bindings/python/Cargo.toml
uv pip install -e dynamo
cargo build --release --manifest-path dynamo/lib/runtime/examples/Cargo.toml -p event_plane_bench
# main sweep (latency vs workers):
python dis2172-bench/run_sweep.py --transports nats,zmq --workers 1,2,4,8,16,32,64,128 \
  --subs 1 --topics kv-events,forward-pass-metrics --duration 15 --warmup 3 --concurrency 32 \
  --outdir dis2172-bench/results/main_sweep
# fan-out sweep (latency vs #subscribers):
python dis2172-bench/run_sweep.py --transports nats,zmq --workers 32 --subs 1,2,4,8 \
  --topics kv-events,forward-pass-metrics --outdir dis2172-bench/results/msweep
python dis2172-bench/plot.py --summary <outdir>/summary.json --outdir <outdir>
```

## 4. Results

### 4.1 Latency vs #workers (M=1) — *measured*
Plots: `results/main_sweep/latency_vs_n_workers_{kv-events,forward-pass-metrics}.png`.
For ZMQ direct mode, **#workers == #ZMQ PUB ports per topic**, so this is also the
"latency vs #ZMQ ports" plot Kyle asked for.

p50 / p99 publish→deliver latency (µs):

| N | KV NATS p50 | KV ZMQ p50 | KV NATS p99 | KV ZMQ p99 | FPM NATS p50 | FPM ZMQ p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 189.1 | 90.7 | 720.4 | 591.2 | 125.7 | 63.1 |
| 2   | 91.0  | 55.2 | 596.5 | 582.1 | 78.2  | 47.2 |
| 4   | 60.7  | 45.6 | 496.7 | 486.5 | 55.2  | 37.4 |
| 8   | 51.5  | 46.2 | 386.5 | 362.3 | 46.7  | 31.3 |
| 16  | 46.5  | 49.4 | 273.3 | 278.3 | 40.4  | 29.8 |
| 32  | 44.8  | 52.8 | 239.4 | 256.9 | 38.5  | 28.8 |
| 64  | 46.6  | 55.2 | 252.2 | 259.7 | 39.7  | 29.6 |
| 128 | 47.2  | 56.8 | 250.2 | 251.5 | 40.1  | 31.4 |

Reading: latency is high at N=1–2 (one/few workers absorb all request load → publisher-side
contention), then flattens once load spreads across workers. In the flat region ZMQ wins on FPM
(~30 vs ~40µs) and trails slightly on KV events (~55 vs ~45µs); both are tens of µs. p99 converges
(~230–260µs) for both at large N.

### 4.2 Brokerless fan-out: connections & drops vs #workers — *measured*
Plot: `results/main_sweep/fanout_vs_n_workers.png`.

| N | NATS est-TCP-conns | ZMQ est-TCP-conns | drops (either) |
|---:|---:|---:|---:|
| 1   | 174 | 154 | 0 |
| 16  | 248 | 310 | 0 |
| 32  | 312 | 438 | 0 |
| 64  | 446 | 681 | 0 |
| 128 | 511 | 864 | 0 |

ZMQ's connection count grows steeply (brokerless direct mode: each subscriber dials every publisher),
NATS grows slowly (single broker connection per process). **No drops** at any N — latency stayed flat
despite the rising connection count. (Counts are a coarse host-wide proxy that also includes request
TCP; the *trend* is the point.)

### 4.3 Fan-out vs #subscribers (M-sweep, N=32) — *measured* — **the decisive result**
Plots/CSV: `results/msweep/` (`latency_vs_n_subs_*.png`, `fanout_vs_n_subs.png`). Fixed N=32
publishers, M ∈ {1,2,4,8} subscribers; p50 averaged across the M subscribers.

| M | KV NATS p50 | KV ZMQ p50 | FPM NATS p50 | FPM ZMQ p50 | NATS conns | ZMQ conns | drops |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46 | 53  | 40 | 30 | 321 | 361  | 0 |
| 2 | 51 | 60  | 43 | 34 | 318 | 557  | 0 |
| 4 | 61 | 76  | 50 | 47 | 249 | 836  | 0 |
| 8 | 84 | **133** | 66 | **82** | 292 | **1284** | 0 |

This is the O(N×M) stress segopal flagged (M router replicas each subscribing to N=32 publishers).
Findings:
- **ZMQ connection count explodes with M** (361→1284 as M:1→8 — the N×M direct mesh), while **NATS
  stays flat** (~300, single broker connection per process). Direct empirical confirmation of
  O(N×M) vs O(N+M).
- **Latency consequence:** ZMQ starts even-or-faster at M=1 (no broker hop) but rises faster than NATS.
  KV-events: ZMQ crosses above NATS by M=2 and is **~1.6× NATS at M=8** (133 vs 84µs). FPM: ZMQ leads
  through M=2, crosses ≈M=4, and trails at M=8 (82 vs 66µs).
- **Still zero drops at M=8 / 1284 connections** — brokerless degrades *gracefully* (latency rises),
  it does **not** cliff-edge or drop at this scale.

**The latency knee is on the subscriber/replica (M) axis, not the publisher (N) axis.**

## 5. Stakeholder asks (DIS-1964) — point-by-point

| Ask (DIS-1964) | Status | Evidence |
|---|---|---|
| Latency vs **#workers** (Kyle) | ✅ measured | §4.1, `latency_vs_workers_*.png` |
| Latency vs **#ZMQ ports** (Kyle) | ✅ measured | §4.1 (#ports == #workers in direct mode) |
| Behavior at scale, **100s of pub-sub connections**, not just 1:1 (segopal) | 🟡 partial | §4.2 (N axis to 128); §4.3 M-sweep to N×M=256 — **single host** |
| **A/B in Astra** (segopal) | ❌ not covered | local single-host only (per "don't pollute my env"); k8s deferred |
| **HW capacity / NIC·CPU saturation** at scale (segopal) | ❌ not covered | loopback host; NIC not exercised — see §6 |
| Multi-level **config clarity** (segopal) | ✅ documented | §2.2 + fairness controls §3.4 |
| Cover all event classes (KV, load, FPM, replica sync) | 🟡 partial | KV events + FPM measured; load-metrics/replica-sync not yet isolated |

## 6. Limitations / not yet covered
- **Single host only** → clean clock (good for latency) but **NIC never exercised**; the multi-host
  NIC/CPU-saturation question (segopal) is unanswered.
- **Single mocker process** runs N workers on one tokio runtime; at very large N, runtime CPU contention
  can mask per-connection cost. A multi-process / multi-host variant is needed for the real fan-out curve.
- **ZMQ broker mode** (XSUB/XPUB, O(N+M)) not yet benchmarked — would need an external proxy.
- **Large payloads**: all traffic here is small (FPM ~0.5 KB, KV batches ~few KB), so the NATS 1 MB cap
  (DGH-892) is never hit; large-embedding / long-context cases not stressed.
- Connection count is a host-wide proxy (includes request-plane TCP), good for trend not absolute.

## 7. Conclusion & recommendation
On a like-for-like single-host comparison:
- **Publisher (N) axis: ZMQ carries no latency penalty** (N→128, M=1: faster on FPM, ~10µs slower on
  KV-events, zero drops). Latency is not a blocker here.
- **Subscriber (M) axis: brokerless cost is real but graceful.** With N=32 and M→8 (256 logical N×M
  connections), ZMQ's connection count explodes (→1284) and its p50 rises above NATS (~1.6× on
  KV-events at M=8), yet it still drops nothing and stays sub-millisecond.

Recommendation:
- **Typical deployments (few router replicas, M≤2–4): default to ZMQ** — competitive or better latency,
  no broker dependency. Supports DYN-2941 for the common case.
- **Many-replica / large fan-out: evaluate ZMQ broker mode (XSUB/XPUB, O(N+M))** before defaulting;
  direct-mode N×M is the dominant cost there.
- **Before flipping the global default**, still required: **multi-host k8s/Astra A/B** (real NIC/CPU
  saturation + network latency) and **large-payload** behavior (NATS 1 MB cap). Those are single-host
  blind spots of this study (§6).

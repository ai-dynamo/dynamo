<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance classification of Dynamo launch settings

Working file for tagging every entry in [dynamo-launch-env-vars.html](dynamo-launch-env-vars.html) as **impact**, **no impact**, or **unexamined**. We work one category at a time; within a category the settings are ordered by how likely they are to move a performance number and by how far they can move it.

## Scoring

Two axes, scored 0-3, multiplied into a rank score. The score orders the table; it is a sorting aid, not the verdict.

**Likelihood (L)** — how often changing this from its default actually moves a performance number in a deployment that is otherwise sensibly configured.

| L | Meaning |
|---|---------|
| 3 | Always. On the request path or the routing decision for every request. |
| 2 | Whenever a common feature is in use (KV routing, disaggregation, multimodal, offload). |
| 1 | Only in a narrow configuration, or the effect is second-order. |
| 0 | Never. Naming, identity, auth, or output formatting only. |

**Magnitude (M)** — how large the swing is when it does bite, on a headline metric (TTFT, ITL, throughput, or frontend CPU).

| M | Meaning |
|---|---------|
| 3 | Large. Tens of percent, or changes the shape of the latency distribution. |
| 2 | Moderate. Roughly 5-20%. |
| 1 | Small. Under about 5%, or only visible under a microbenchmark. |
| 0 | None measurable. |

**Verdict** — `impact` for L x M >= 4 or any entry where a plausible misconfiguration is costly; `no impact` for the rest.

**Deprecated settings** are judged on what the code does with them, not on their status. A setting that is deprecated but still read and still changes behaviour keeps the verdict its effect earns. A setting that is accepted and then discarded — parsed only so old launch commands keep starting — is `no impact`, because setting it changes nothing. Both kinds appear below.

## Status and workflow

**Nothing here is applied.** Every verdict below is a proposal awaiting human validation, and [dynamo-launch-env-vars.html](dynamo-launch-env-vars.html) deliberately still shows all 411 settings as `unexamined`.

The review loop:

1. This file is the authored source. [render_classification.py](render_classification.py) turns it into [perf-classification-review.html](perf-classification-review.html), which presents each proposal next to the setting's flag, default, and component scope.
2. A human works through that page, agreeing with a verdict or flipping it. Decisions persist in the browser, and the page exports them as a ready-to-paste `PERF` dict.
3. Only then does that dict go into the `PERF` map in [generate.py](generate.py) and `python3 notes/env-vars/generate.py` rebuild the catalogue page with real tags.

Keys are the variable name, or the flag for a CLI-only row. Anything absent from `PERF` stays `unexamined`. A variable is one record, so one verdict covers every tab it appears in.

## Plan

Categories are listed in the order they were worked: within each group, by how much performance judgement the category needs. All 38 are complete. Shared settings — those read by more than one component — are ranked once here, under **Shared**, rather than repeated under each group.

| # | Group | Category | Entries | Status |
|---|-------|----------|--------:|--------|
| 1 | Frontend | KV router tuning | 32 | proposed |
| 2 | Frontend | Router: mode & admission | 8 | proposed |
| 3 | Frontend | Preprocessing, templates & parsers | 7 | proposed |
| 4 | Frontend | HTTP service & API surface | 27 | proposed |
| 5 | Frontend | Frontend core | 23 | proposed |
| 6 | Frontend | Metrics | 7 | proposed |
| 7 | Frontend | LoRA | 6 | proposed |
| 8 | Frontend | AIC performance model | 11 | proposed |
| 9 | Frontend | Multimodal | 1 | proposed |
| 10 | Frontend | CLI-only flags | 5 | proposed |
| 11 | Shared | Request / event plane | 19 | proposed |
| 12 | Shared | KVBM (KV block manager) | 20 | proposed |
| 13 | Shared | Worker runtime & identity | 21 | proposed |
| 14 | Shared | Request tracing | 24 | proposed |
| 15 | Shared | Tokio runtime | 4 | proposed |
| 16 | Shared | Forward-pass metric trace | 6 | proposed |
| 17 | Shared | Health checks | 7 | proposed |
| 18 | Shared | Topology & KV transfer | 5 | proposed |
| 19 | Shared | Multimodal HTTP fetch client | 7 | proposed |
| 20 | Shared | Multimodal | 4 | proposed |
| 21 | Shared | Model download | 10 | proposed |
| 22 | Shared | Logging | 6 | proposed |
| 23 | Shared | OpenTelemetry export | 9 | proposed |
| 24 | Shared | NATS | 8 | proposed |
| 25 | Shared | etcd | 7 | proposed |
| 26 | Shared | System status server | 6 | proposed |
| 27 | Shared | Shutdown & lifecycle | 3 | proposed |
| 28 | Shared | Discovery | 2 | proposed |
| 29 | Shared | Profiling | 2 | proposed |
| 30 | Shared | Memory | 1 | proposed |
| 31 | Shared | LoRA / RL / Frontend core / KV router / CLI-only | 9 | proposed |
| 32 | vLLM | vLLM engine wrapper | 21 | proposed |
| 33 | vLLM | vLLM extras (non-CLI) | 8 | proposed |
| 34 | vLLM | LoRA + Multimodal | 4 | proposed |
| 35 | TensorRT-LLM | TensorRT-LLM engine wrapper | 49 | proposed |
| 36 | TensorRT-LLM | extras, KVBM, Multimodal | 5 | proposed |
| 37 | SGLang | SGLang engine wrapper | 12 | proposed |
| 38 | SGLang | SGLang extras (non-CLI) + Multimodal | 5 | proposed |

411 entries total: frontend 127, shared 180, vLLM 33, TensorRT-LLM 54, SGLang 17.

---

## 1. Frontend — KV router tuning (32)

Everything here is inert unless the router is actually running the KV policy: `--router-mode kv`, or `--load-aware`, which implies it. Under any other router mode these settings are parsed and ignored, so read every row below as "given KV routing is on". That precondition is itself `DYN_ROUTER_MODE`, which belongs to category 2.

The dominant effect in this category is on **TTFT**, through prefix cache hit rate — a routing decision that lands on a worker already holding the prefix skips prefill entirely. The second effect is on **frontend CPU**, through the indexer that maintains the radix tree.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_ROUTER_USE_KV_EVENTS` | 3 | 3 | 9 | impact | Switches the router between exact cache state fed by worker events and cache state predicted from its own routing decisions. Directly sets prefix hit rate, and therefore TTFT, on every request. Also changes the event-plane traffic the frontend must consume. |
| 2 | `DYN_ROUTER_LOAD_AWARE` | 3 | 3 | 9 | impact | A preset, not a knob: sets `overlap_score_credit=0`, disables KV events and KV-reuse assumption, enables active-block and prefill-token tracking, disables remote/shared indexers, and forces `--router-mode kv`. Flips the router from cache-affinity to pure load balancing in one flag. |
| 3 | `DYN_ROUTER_PREFILL_LOAD_SCALE` | 3 | 3 | 9 | impact | The scale on adjusted prefill load after overlap credits are subtracted — one of the two terms in the routing cost function. Sets how hard cache affinity is traded against load. Every request is scored with it. |
| 4 | `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` | 3 | 3 | 9 | impact | The other term: credit for device-local prefix overlap. Above 1.0 the adjusted cost can go negative, so the router will pile onto a cache-warm worker past the point where load should win. |
| 5 | `DYN_ROUTER_QUEUE_THRESHOLD` | 3 | 3 | 9 | impact | Unset means no router-side queueing at all. Setting it turns on queueing when every worker exceeds the fraction of `max_num_batched_tokens` — changes the whole shape of the TTFT distribution under load, not just its mean. |
| 6 | `DYN_ROUTER_TRACK_ACTIVE_BLOCKS` | 3 | 3 | 9 | impact | On by default. Turning it off removes the load signal the router balances on, leaving overlap alone to pick workers. Large regression in load spread on heterogeneous traffic. |
| 7 | `DYN_USE_REMOTE_INDEXER` | 2 | 3 | 6 | impact | Experimental. Moves the radix tree off the frontend to a worker-served indexer queried over the request plane: trades frontend CPU and memory for a per-request RPC on the routing path. Both sides of that trade are large. |
| 8 | `DYN_ROUTER_EVENT_THREADS` | 2 | 3 | 6 | impact | Indexer worker threads; above 1 switches to a concurrent radix tree with a thread pool. This is the frontend's KV-indexing throughput ceiling and a direct frontend-CPU knob — the main reason this category matters to frontend benchmarking. |
| 9 | `DYN_ROUTER_TRACK_PREFILL_TOKENS` | 2 | 3 | 6 | impact | On by default. Excluding prompt tokens from load accounting changes routing, queue pressure, and the `active_prefill_tokens` metric together — long-prompt traffic is the case where it bites hardest. |
| 10 | `DYN_SHARED_CACHE_TYPE` | 2 | 3 | 6 | impact | Experimental. `hicache` makes the router consult a Mooncake master for SGLang L3 state on the routing path: adds an external lookup per decision, and adds a whole cache tier to the hit-rate calculation. |
| 11 | `DYN_ROUTER_TEMPERATURE` | 2 | 2 | 4 | impact | Softmax sampling over worker scores. Non-zero deliberately trades prefix locality for spread; the cache hit rate falls as temperature rises. |
| 12 | `DYN_ROUTER_ASSUME_KV_REUSE` | 2 | 2 | 4 | impact | On by default. Turning it off makes the router generate random block hashes, so active-block tracking no longer collapses shared prefixes — correct when the engine has reuse disabled, costly when it does not. |
| 13 | `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` | 2 | 2 | 4 | impact | Adds placeholder blocks during generation with decay against expected output length. Improves decode-side load estimates on long generations; adds per-token router bookkeeping. |
| 14 | `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY` | 2 | 2 | 4 | impact | Decays overlap credit as a worker's excess prefill load grows — the guard that stops cache affinity from overloading one worker. Matters exactly at the load levels benchmarks run at. |
| 15 | `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT` | 2 | 2 | 4 | impact | Experimental. Adds a per-active-request decode cost, for engines where step latency tracks request count rather than KV footprint. Changes decode batch balance when non-zero. |
| 16 | `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` | 2 | 2 | 4 | impact | Credit for host-pinned (CPU offload) overlap. Inert without KVBM host offload; with it, sets how strongly the router prefers a worker that must stage KV back from host. |
| 17 | `DYN_ROUTER_DISK_CACHE_HIT_WEIGHT` | 2 | 2 | 4 | impact | Same as above for the disk tier. Lower default (0.25) because the restore is slower; wrong value sends requests to workers whose "hit" costs more than a fresh prefill. |
| 18 | `DYN_ROUTER_POLICY_CONFIG` | 2 | 2 | 4 | impact | Startup-only YAML defining policy families and cache-bucket queues. When present it supersedes the single synthetic policy class built from threshold and policy, so it can restructure queueing entirely. |
| 19 | `DYN_ROUTER_QUEUE_POLICY` | 2 | 2 | 4 | impact | Only live once queueing is enabled. `fcfs` optimizes tail TTFT, `wspt` optimizes mean TTFT — it moves the metric you are measuring, which is exactly the trap in an A/B. |
| 20 | `DYN_ROUTER_TTL_SECS` | 2 | 2 | 4 | impact | Block TTL in approximate mode only (`--no-router-kv-events`). Sets how long the router believes in cache state it never had confirmed; too long routes to evicted prefixes, too short discards good ones. |
| 21 | `DYN_ROUTER_PREDICTED_TTL_SECS` | 2 | 2 | 4 | impact | Enables predict-on-route entries in the local side indexer, closing the window between a routing decision and the worker's KV event. Helps most on bursty traffic with short prompts. |
| 22 | `DYN_USE_KV_EVENTS` | 1 | 3 | 3 | impact | Legacy alias for `DYN_ROUTER_USE_KV_EVENTS`. Rarely set now, but when set its effect is identical to entry 1 — the magnitude is the same, only the likelihood is lower. |
| 23 | `DYN_SHARED_CACHE_MULTIPLIER` | 1 | 2 | 2 | impact | Discounts a shared-cache hit against a device-local one. Inert unless `DYN_SHARED_CACHE_TYPE` is set, but then it is the term that decides whether shared hits are worth routing for. |
| 24 | `DYN_ROUTER_PREFILL_LOAD_MODEL` | 1 | 2 | 2 | impact | Experimental. `aic` decays the oldest active prefill request by AIC-predicted duration instead of static accounting. Needs the AIC database configured, so rarely on, but it changes the load term when it is. |
| 25 | `DYN_ROUTER_REPLICA_SYNC` | 1 | 2 | 2 | impact | Best-effort active-sequence sync across router replicas over the event plane. Only meaningful with several frontends; improves their combined load view at the cost of event-plane traffic. |
| 26 | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` | 1 | 2 | 2 | impact | Deprecated and warned, but still read: it sets the legacy overlap weight and so still perturbs the cost function on any deployment that has not migrated. |
| 27 | `DYN_OVERLAP_SCORE_WEIGHT` | 1 | 2 | 2 | impact | Older alias of the above, read only when that one is unset. Same reasoning, one step further from current practice. |
| 28 | `DYN_ENCODER_CUDA_TO_CPU_RATIO` | 1 | 2 | 2 | impact | Weighting for device-aware-weighted routing across mixed CUDA and CPU encode workers. Narrow — multimodal encode fleets only — but within that fleet it sets the work split directly. |
| 29 | `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS` | 1 | 1 | 1 | no impact | A cleanup guard for stale active-request entries, not a request timeout. Only matters if entries leak; on a healthy router it never fires. |
| 30 | `DYN_ROUTER_TRACKING_HASH` | 1 | 1 | 1 | no impact | `keyed-xxh3-v1` costs marginally more than `public-xxh3-v1` per identity, on a path already dominated by tokenization. Chosen for privacy, not speed. |
| 31 | `DYN_ROUTER_TRACKING_KEY_FILE` | 0 | 0 | 0 | no impact | Supplies the 32-byte key for keyed tracking. Identity material; no path cost. |
| 32 | `DYN_ROUTER_TRACKING_KEY_ID` | 0 | 0 | 0 | no impact | Key epoch identifier carried alongside the key. Metadata only. |

**Summary:** 28 `impact`, 4 `no impact`.

Entries 22, 26, and 27 are deprecated but still read and still effective, so they keep the verdict their effect earns. Contrast entries 7 and 8 of category 2, which are deprecated *and discarded*.

---

## 2. Frontend — Router: mode & admission (8)

This category holds the switch that decides whether category 1 is live at all, plus the worker-busy rejection checks. The rejection thresholds are the only settings in the frontend that can turn a slow request into a refused one, so their failure mode is a change in success rate rather than in latency — worth separating in any measurement.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_ROUTER_MODE` | 3 | 3 | 9 | impact | Selects the routing algorithm outright, and is the precondition for all 32 settings in category 1 — only `kv` makes them live. Beyond that it carries a cliff of its own: under disaggregated prefill, `power-of-two` and `least-loaded` skip the bootstrap optimization and fall back to the synchronous prefill path, so the mode changes the disagg data path and not just worker choice. The single most consequential frontend setting. |
| 2 | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC` | 2 | 3 | 6 | impact | Unset means the check is off entirely. Setting it marks a worker busy above a fraction of `max_num_batched_tokens`, which gates admission — under load this converts queueing or slow service into rejection. Changes the success-rate axis, not only latency. |
| 3 | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD` | 2 | 3 | 6 | impact | The absolute-token form of the same check, OR-ed with the fractional one. Same effect and same failure mode; easier to misjudge because it is not relative to the engine's batch budget, so a value tuned for one `max_num_batched_tokens` is wrong for another. |
| 4 | `DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD` | 2 | 3 | 6 | impact | The decode-side counterpart: busy above a fraction of KV block utilization. Off by default. Its natural setting is close to where the engine would start preempting anyway, so it trades preemption for rejection — a real choice, with a large effect on tail behaviour. |
| 5 | `DYN_ROUTER_SESSION_AFFINITY_TTL_SECS` | 2 | 2 | 4 | impact | Off unless set. Pinning a session to a worker for an idle TTL raises prefix hit rate on multi-turn traffic and works against load spread at the same time. Which effect wins depends on the workload, so it needs measuring rather than assuming. |
| 6 | `DYN_ROUTER_MIN_INITIAL_WORKERS` | 2 | 2 | 4 | impact | Zero by default, meaning the router starts serving before any worker has registered. Steady-state effect is nil, but it decides whether the opening seconds of a run are measured against a partial fleet — which is exactly the window a benchmark's warmup is meant to exclude. Tagged for measurement validity, not for steady-state cost. |
| 7 | `DYN_ADMISSION_CONTROL` | 0 | 0 | 0 | no impact | The legacy master switch. Still accepted so existing launch commands keep starting, but it warns and sets nothing on the namespace; the explicit thresholds in rows 2-4 are what actually gate admission. Setting it changes nothing. |
| 8 | `DYN_ENFORCE_DISAGG` | 0 | 0 | 0 | no impact | Deprecated and ignored: routing topology and readiness are derived from the registered worker types. Accepted for compatibility only. |

**Summary:** 6 `impact`, 2 `no impact`. The CLI-only `--admission-control` flag carries the same `no impact` verdict as row 7.

---

## 3. Frontend — Preprocessing, templates & parsers (7)

This is the frontend's own CPU, not the workers'. Tokenization, template rendering, and per-chunk parsing are what the frontend process actually spends its cycles on, so every entry here lands on throughput and on frontend CPU per request rather than on engine time.

Four settings that belong to this story sit in **Frontend core** (category 5) because that is where their arg group puts them: `DYN_TOKENIZER` (HuggingFace vs fastokens backend), `DYN_CHAT_PROCESSOR` (Rust preprocessor vs local vLLM/SGLang), `DYN_PREPROCESS_WORKERS` (offload preprocessing to worker processes), and `DYN_DEBUG_PERF`. They are the largest knobs on this path; they get judged in category 5, and rows 2, 3, and 6 below are only live for particular values of `DYN_CHAT_PROCESSOR`.

The tokenizer cache is unusual in being directly instrumented — `dynamo_frontend_tokenizer_cache_cached_tokens_total` against `..._uncached_tokens_total` gives the hit rate straight from the metrics endpoint, so rows 1, 4, and 5 can be measured rather than argued about.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_TOKENIZER_CACHE` | 3 | 3 | 9 | impact | On unless set to `0`. Disabling it makes every request re-tokenize its full prompt on the frontend — the dominant frontend CPU cost, and the one that scales with prompt length. On repetitive or long-prefix traffic this is the difference between a cheap lookup and a full BPE pass per request. |
| 2 | `DYN_VLLM_STREAM_INTERVAL` | 2 | 3 | 6 | impact | Token batching interval for the vLLM pre/post processor path. Sets how many tokens are coalesced before an SSE flush, trading measured ITL granularity against per-chunk frontend work. Live only under `--dyn-chat-processor vllm`, but decisive there — it moves both the cost and the metric. |
| 3 | `DYN_SGLANG_STREAM_INTERVAL` | 2 | 3 | 6 | impact | The same knob on the SGLang processor path, with the same trade and the same caveat that it is inert unless that processor is selected. |
| 4 | `DYN_TOKENIZER_CACHE_BYTES` | 2 | 3 | 6 | impact | 64 MiB by default. The budget sets the hit rate, and hit rate is what row 1 is worth — sized below the working set, the cache thrashes and degrades toward the disabled case while still paying insert cost. Note it silently falls back to the default on an unparseable value. |
| 5 | `DYN_TOKENIZER_CACHE_EXTEND` | 2 | 2 | 4 | impact | Partial-hit extension, on unless set to `0`. Disabling it means prompts that share a prefix with a cached entry stop contributing new suffixes, so the cache stays cold on evolving conversations — the exact traffic shape where prefix reuse should pay. |
| 6 | `DYN_VLLM_SKIP_REQUEST_VALIDATION` | 2 | 2 | 4 | impact | Defaults to `1`, i.e. validation skipped. Turning it on adds per-request re-validation in the Python pre/post path. Bounded work, but it is per request and it is Python, so it shows up in frontend CPU under concurrency. Only live on the Python processor paths. |
| 7 | `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` | 1 | 2 | 2 | impact | Swaps the tool-call parser implementation for supported families (Qwen3-Coder, DeepSeek-V4) on both batch and streaming paths, so a different parser runs on every streaming chunk. Primarily a correctness and behaviour switch — it changes truncation semantics at EOF — with the CPU effect as a side effect. The weakest `impact` call in this category; happy to move it if you would rather the tag track intent than measurable cost. |

**Summary:** 7 `impact`, 0 `no impact`.

An all-`impact` category is the honest outcome here: nothing on this list is naming or identity, every entry sits on the per-request frontend path, and the two that are narrowest (rows 6 and 7) are narrow because of a precondition, not because their effect is small.

---

## 4. Frontend — HTTP service & API surface (27)

The opposite shape to category 3: a handful of real timing knobs sitting in a large pile of route naming and feature registration. Eleven of the 27 are `DYN_HTTP_SVC_*_PATH` overrides that only change the string a route is mounted at, and several others just register an endpoint. Those are grouped at the bottom rather than argued one by one.

Two entries earn `impact` for a reason worth stating up front, because it is not latency: `DYN_HTTP_OVERLOAD_STATUS_CODE` changes what a load generator does when it is refused, and `DYN_MAX_OUTPUT_TOKENS` changes the workload's own shape. Both move the number you measure without touching the code path being measured.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS` | 2 | 3 | 6 | impact | Inactivity circuit breaker: kills the engine context and drops the inflight guard when no SSE event arrives within the window. Unset, a zombie worker holding a live TCP connection stalls the request indefinitely and keeps its slot — the classic cause of a benchmark's tail running off the end of the chart. Set too tight, healthy slow requests get killed. |
| 2 | `DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS` | 2 | 3 | 6 | impact | Off unless set. When on, the frontend polls the engine stream for a synchronous error before committing HTTP 200, spending up to the window on first-token latency to convert SSE error frames into HTTP 4xx. The guidance is to set it at or above the observed parse/admission p99, so by construction it is sized in the same units as TTFT and paid on slow-start requests. |
| 3 | `DYN_MAX_OUTPUT_TOKENS` | 2 | 3 | 6 | impact | A server-side ceiling on `max_tokens`. Not a code-path change — a workload change: it silently truncates the output length of every request that asks for more, so decode work per request, tokens/s, and any OSL-derived metric all shift. A run whose OSL is being clamped without the operator knowing is measuring a different workload than intended. |
| 4 | `DYN_STRIP_ANTHROPIC_PREAMBLE` | 2 | 3 | 6 | impact | Removes the Claude Code billing preamble from the system prompt. Fewer prompt tokens on every affected request, and — because the preamble sits at the front — a materially better prefix-cache hit rate downstream. Narrow to that traffic, decisive within it; the help text names prompt caching as the point. |
| 5 | `DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS` | 2 | 2 | 4 | impact | Disabled unless set to a positive value. Emits an SSE comment on an idle stream, which costs a timer and a write per interval per in-flight stream. At high streaming concurrency with a short interval this is real frontend work; its purpose is to stop intermediaries dropping idle connections, which is itself a tail-latency concern. |
| 6 | `DYN_CONTEXT_WINDOW` | 2 | 2 | 4 | impact | Overrides the advertised context window used for request validation. Raising it lets longer requests through to the engine, lowering it rejects them at the frontend — either way it moves where the ISL distribution gets cut, and the Anthropic model-info route reports it. |
| 7 | `DYN_HTTP_OVERLOAD_STATUS_CODE` | 1 | 3 | 3 | impact | Defaults to 529. Setting 503 is explicitly for retry semantics: most load generators and gateways retry a 503 and do not retry a 529. That flips a refusal from a recorded failure into additional offered load, changing measured throughput and success rate without any code path differing. Tagged for that, not for the cost of returning a number. |
| 8 | `DYN_DISABLE_FRONTEND_NVEXT` | 1 | 2 | 2 | impact | Disables the `nvext` protocol, which also makes the frontend ignore the routing-override headers. Any harness that pins requests to workers through those headers silently loses that control and falls back to normal routing — a routing change disguised as a protocol switch. Tagged for the override headers, not for dropping `request.nvext`. |
| 9 | `DYN_HTTP_BODY_LIMIT_MB` | 1 | 1 | 1 | no impact | Gates whether a request is accepted at all. Large multimodal payloads above the limit are rejected outright, which invalidates a run rather than slowing it; accepted requests are unaffected. A capacity gate, not a performance knob. |
| 10 | `DYN_ENABLE_STREAMING_TOOL_DISPATCH` | 1 | 1 | 1 | no impact | Emits one extra SSE event per completed tool call. A handful of additional writes per response on tool-calling traffic — genuinely smaller than the parser swap in category 3 row 7, which changes the algorithm running on every chunk. |
| 11 | `DYN_DISABLE_FRONTEND_ADMIN_API` | 1 | 1 | 1 | no impact | Unregisters `GET`/`POST /busy_threshold`. No steady-state cost, but worth knowing it removes the ability to retune the busy threshold at runtime — the knob goes away, not the performance. |
| 12 | `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | 1 | 1 | 1 | no impact | Drain window before forced shutdown. Costs wall-clock between runs in a paired A/B with many restarts; does not change any number measured during a run. |
| 13 | `DYN_ENABLE_FORCE_INCLUDE_USAGE` | 1 | 0 | 0 | no impact | Forces a usage block into streaming responses regardless of `stream_options.include_usage`. One extra final chunk. |
| 14 | `DYN_METADATA_HEADER` | 0 | 0 | 0 | no impact | Names the header carrying opaque per-request metadata to workers. Propagation only. |
| 15 | `DYN_ENABLE_ANTHROPIC_API` | 0 | 0 | 0 | no impact | Registers `/v1/messages`. Serving traffic on that route exercises the Anthropic path, but the switch itself only mounts it. |
| 16 | `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE` | 0 | 0 | 0 | no impact | Registers the vLLM-compatible `/inference/v1/generate` route. Same reasoning. |
| 17-27 | `DYN_HTTP_SVC_*_PATH` (11) | 0 | 0 | 0 | no impact | Route path strings for chat, completions, embeddings, responses, Anthropic messages, models, files, batches, metrics, health, and live. They change where a handler is mounted, never what it does. |

**Summary:** 8 `impact`, 19 `no impact`.

Rows 7 and 8 are the judgement calls. Both are tagged on an indirect effect — a client's retry behaviour and a harness's routing overrides respectively — rather than on frontend cost. If the tag should mean "changes what this process does per request", both drop to `no impact`; I have read it as "changes a performance number you would measure", which they do.

---

## 5. Frontend — Frontend core (23)

The catch-all group of the frontend's own arg parser, and it holds the three largest knobs on the whole frontend CPU path — the ones category 3 deferred here. It also forces a distinction that has been implicit until now.

**Subject versus setting.** `DYN_MODEL_PATH` and `DYN_MODEL_NAME` obviously determine the numbers a run produces, but they define *what is being measured*, not how it performs. The convention I have applied: a setting that silently changes a workload the operator asked for is `impact` (category 4's `DYN_MAX_OUTPUT_TOKENS` clamping `max_tokens` is the example); a setting that names the subject under test is `no impact`. Filtering the page to `impact` should surface knobs to tune, not the identity of the thing being tuned.

Four settings shown by the HTML in this section — `DYN_NAMESPACE`, `DYN_DUMP_CONFIG_TO`, `DYN_REQUEST_PLANE`, `DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE` — are read by all four components and are ranked once in the Shared group, not here.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_PREPROCESS_WORKERS` | 3 | 3 | 9 | impact | 0 by default, meaning tokenization, template rendering, and detokenization all run on the main event loop. Above 0 they move to a ProcessPoolExecutor where each worker has its own GIL. This is the frontend's answer to being single-GIL-bound, so it is the first knob to reach for when the frontend is the bottleneck and the one most likely to change a throughput number outright. |
| 2 | `DYN_CHAT_PROCESSOR` | 3 | 3 | 9 | impact | Chooses between the Rust preprocessor and doing pre/post processing in local vLLM or SGLang Python. A different implementation in a different language on the per-request path — not a tuning parameter but a swap of the component under test. It also decides whether category 3 rows 2, 3, and 6 are live at all. |
| 3 | `DYN_TOKENIZER` | 3 | 3 | 9 | impact | `default` (HuggingFace tokenizers) against `fastokens`, whose stated purpose is high-performance BPE encoding. Encoding is the frontend's dominant per-request cost, so this is a direct swap of the hottest component. Two caveats that bound it: decoding always uses HuggingFace, and TikToken models are unaffected. |
| 4 | `DYN_KV_CACHE_BLOCK_SIZE` | 2 | 3 | 6 | impact | The block granularity the router hashes and indexes prefixes at. It must agree with the engine's block size; when it does not, prefix matching degrades or fails silently and the KV router's hit rate collapses while every metric still looks plausible. Sharp edge rather than a dial. |
| 5 | `DYN_FRONTEND_FD_LIMIT_TARGET` | 2 | 3 | 6 | impact | Raises `RLIMIT_NOFILE` at startup. Every client connection and every worker connection is a descriptor, so a target below what the concurrency level needs turns into accept failures and connection errors at exactly the load you were trying to measure. A ceiling, and ceilings bind hardest under benchmark conditions. |
| 6 | `DYN_MIGRATION_LIMIT` | 2 | 3 | 6 | impact | 0 by default, which disables migration. Above 0, a request whose worker disconnects is retried on another worker — which re-runs prefill from scratch. It converts a failure into a very slow success, so it moves the tail rather than the median, and it changes the failure count a run reports. |
| 7 | `DYN_SERVE_INDEXER` | 2 | 3 | 6 | impact | Serves this frontend's KV indexers over the request plane for other frontends to query — the supply side of category 1's `DYN_USE_REMOTE_INDEXER`. Enabling it puts indexer query load on this process on top of its own serving work. |
| 8 | `DYN_TLS_CERT_PATH` | 2 | 3 | 6 | impact | Supplying a certificate switches the server to HTTPS, adding a handshake per connection and encryption per byte on the frontend's own CPU. Streaming responses pay it continuously. A plaintext-to-TLS comparison is not a like-for-like frontend measurement. |
| 9 | `DYN_TLS_KEY_PATH` | 2 | 3 | 6 | impact | The other half of the same switch; same reasoning, and TLS only engages when both are present. |
| 10 | `DYN_MIGRATION_MAX_SEQ_LEN` | 2 | 2 | 4 | impact | Caps the sequence length for which migration state is retained. Unset means no limit, so the frontend caches migration state for arbitrarily long sequences — the help text calls out unbounded memory growth as the reason it exists. Trades frontend memory against how much of the traffic stays migratable. |
| 11 | `DYN_DEBUG_PERF` | 2 | 2 | 4 | impact | Per-function timing and hot-path section durations on the preprocessing path. Instrumentation on the hot path costs something to collect — the observer effect is the point of tagging it, since it is switched on precisely when someone is measuring. |
| 12 | `DYN_KSERVE_GRPC_SERVER` | 1 | 2 | 2 | impact | Starts a second serving stack, with its own threads, inside the same process. Idle it is cheap; carrying traffic it competes with the HTTP path for the same cores. |
| 13 | `DYN_FRONTEND_ROUTE_EXTENSIONS` | 1 | 1 | 1 | no impact | Loads trusted route extensions by entry-point name or `module:function`. Whatever an extension costs belongs to the extension, not to this flag — the flag only mounts it. |
| 14 | `DYN_ENABLE_STREAMING_REASONING_DISPATCH` | 1 | 1 | 1 | no impact | One additional SSE event per response once thinking ends. Same reasoning as category 4 row 10. |
| 15 | `DYN_TRUST_REMOTE_CODE` | 1 | 1 | 1 | no impact | Permits model-supplied tokenizer code to load. It decides whether some models start at all, which is a capability gate rather than a cost; the resulting tokenizer's speed is the model's property. |
| 16 | `DYN_NAMESPACE_PREFIX` | 1 | 0 | 0 | no impact | Widens model discovery to every namespace sharing a prefix. Discovery-time scoping; the request path does not consult it. |
| 17 | `DYN_INTERACTIVE` | 0 | 0 | 0 | no impact | Runs a terminal chat instead of serving HTTP. It replaces the mode of operation rather than tuning it — there is no serving path to be fast or slow. |
| 18 | `DYN_MODEL_PATH` | 0 | 0 | 0 | no impact | Names the model the frontend loads its card and tokenizer from. It defines the subject under test; see the note above. |
| 19 | `DYN_MODEL_NAME` | 0 | 0 | 0 | no impact | The served name for that model. Identity only. |
| 20 | `DYN_HTTP_PORT` | 0 | 0 | 0 | no impact | Listen port. |
| 21 | `DYN_HTTP_HOST` | 0 | 0 | 0 | no impact | Bind address. |
| 22 | `DYN_GRPC_METRICS_PORT` | 0 | 0 | 0 | no impact | Metrics port for the gRPC service, used only alongside row 12. |
| 23 | `DYN_RL_PORT` | 0 | 0 | 0 | no impact | Port for the RL weight-sync router. Binding only; whether RL is on is a different setting. |

**Summary:** 12 `impact`, 11 `no impact`.

Rows 1 to 3 are the three settings I would check first on any frontend measurement — together they decide the language, the process topology, and the tokenizer implementation of the entire per-request path. Row 4 is the one most likely to be wrong without anyone noticing.

---

## 6. Frontend — Metrics (7)

Six histogram bucket configurations and one name prefix. These do not make the frontend faster or slower in any meaningful way — they decide **what number you read back**. Prometheus quantiles are interpolated from bucket boundaries, so a TTFT histogram whose buckets are too coarse around the real distribution reports a p99 that is simply wrong. That is why the six are tagged, on the same measurement-fidelity grounds as `DYN_ROUTER_MIN_INITIAL_WORKERS` in category 2. If the tag should mean *the system's own speed*, all six flip to `no impact` together; they are the cleanest single group to reverse.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_METRICS_TTFT` | 2 | 3 | 6 | impact | Buckets for the metric most perf work is judged on. Default `0.001,480.0,18` spans three orders of magnitude in 18 buckets, so resolution near any particular TTFT is coarse — reported quantiles move when you change it, without the system changing at all. |
| 2 | `DYN_METRICS_ITL` | 2 | 3 | 6 | impact | Same for inter-token latency, where the distribution is narrow and the default span is wide — the setting most likely to be quietly under-resolved. |
| 3 | `DYN_METRICS_REQUEST_DURATION` | 2 | 2 | 4 | impact | Buckets for end-to-end duration. Wide spans are more forgiving here, but the same interpolation caveat applies to any quantile read off it. |
| 4 | `DYN_METRICS_INPUT_SEQUENCE` | 1 | 2 | 2 | impact | ISL histogram. Governs how faithfully the recorded workload shape matches the offered one. |
| 5 | `DYN_METRICS_OUTPUT_SEQUENCE` | 1 | 2 | 2 | impact | OSL histogram, same reasoning. Worth checking whenever OSL is a headline of the run. |
| 6 | `DYN_METRICS_EMBEDDING_LATENCY` | 1 | 2 | 2 | impact | Embedding latency histogram; only populated on embedding traffic. |
| 7 | `DYN_METRICS_PREFIX` | 0 | 0 | 0 | no impact | Renames the metric family from `dynamo_frontend`. Dashboards break, numbers do not move. |

**Summary:** 6 `impact`, 1 `no impact`.

---

## 7. Frontend — LoRA (6)

The LoRA allocation controller: which adapters live on which workers, recomputed on a timer. Entirely inert unless row 1 is on, but once it is, every entry here is a tuning parameter of a placement algorithm, and placement decides how often a request arrives at a worker that does not hold its adapter — the LoRA equivalent of a prefix-cache miss.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_LORA_ALLOCATION_ENABLED` | 2 | 3 | 6 | impact | The master switch. Off, adapters are placed by whatever the default path does; on, a controller actively reallocates them. Everything below is dead until this is set. |
| 2 | `DYN_LORA_ALLOCATION_ALGORITHM` | 2 | 3 | 6 | impact | `hrw`, `random`, or `mcf`. Rendezvous hashing, uniform random, and a min-cost-flow solver produce genuinely different adapter locality — `random` is the control arm, not a serious operating choice. |
| 3 | `DYN_LORA_MCF_CONFIG` | 2 | 2 | 4 | impact | The MCF solver's cost terms (`gamma_load`, `beta_keep`, `candidate_m`). `beta_keep` in particular sets how strongly the solver avoids moving an adapter, which is the churn-versus-balance trade in one number. |
| 4 | `DYN_LORA_ALLOCATION_TIMESTEP_SECS` | 2 | 2 | 4 | impact | Recompute interval. Short reacts to load shifts and risks thrashing adapters between workers; long leaves placement stale through a traffic change. |
| 5 | `DYN_LORA_ALLOCATION_PREDICTOR_TYPE` | 2 | 2 | 4 | impact | `none` feeds raw counts to the solver, `ema` feeds a smoothed estimate. Determines whether placement chases noise. |
| 6 | `DYN_LORA_ALLOCATION_EMA_ALPHA` | 1 | 2 | 2 | impact | Smoothing factor for that EMA. Only live under `ema`, where it sets how much history the load estimate carries. |

**Summary:** 6 `impact`, 0 `no impact`.

---

## 8. Frontend — AIC performance model (11)

Inputs to an AIConfigurator latency lookup, used by the mocker and by `--router-prefill-load-model aic`. Nothing here executes inference; it parameterises a *prediction* of inference. That makes the whole category conditional — inert unless one of those two consumers is active — but when it is active, a wrong parameter means the router is scheduling against a model of a machine it is not running on. The likelihood scores are low throughout for that reason and the verdicts are driven by the cost of getting them wrong, not by how often they are set.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_AIC_SYSTEM` | 1 | 3 | 3 | impact | Names the hardware to model, e.g. `h200_sxm`. Predicting H200 latencies on H100 silicon skews every routing decision the load model informs, and nothing in the system will contradict it. |
| 2 | `DYN_AIC_BACKEND` | 1 | 3 | 3 | impact | Backend family to model (vllm, sglang). Wrong family, wrong latency curve, same silent failure. |
| 3 | `DYN_AIC_MODEL_PATH` | 1 | 3 | 3 | impact | Model identifier for the lookup. Must describe the model actually being served for the predictions to mean anything. |
| 4 | `DYN_AIC_TP_SIZE` | 1 | 3 | 3 | impact | Tensor-parallel size to model. Latency scales strongly with it, so a mismatch against the real deployment misprices every request. |
| 5 | `DYN_AIC_NEXTN` | 1 | 3 | 3 | impact | MTP/Eagle draft-token count. Speculative decoding changes the shape of decode latency, not just its scale; omitting it models a non-speculative engine. |
| 6 | `DYN_AIC_NEXTN_ACCEPT_RATES` | 1 | 2 | 2 | impact | Conditional accept rates for those draft tokens. They set the expected acceptance and therefore the predicted speedup; padded or truncated to `--aic-nextn`. |
| 7 | `DYN_AIC_ATTENTION_DP_SIZE` | 1 | 2 | 2 | impact | Attention data-parallel size in the model. Affects predicted attention cost. |
| 8 | `DYN_AIC_MOE_TP_SIZE` | 1 | 2 | 2 | impact | MoE tensor-parallel size, required by some MoE models for the lookup to resolve at all. |
| 9 | `DYN_AIC_MOE_EP_SIZE` | 1 | 2 | 2 | impact | MoE expert-parallel size, same. |
| 10 | `DYN_AIC_BACKEND_VERSION` | 1 | 2 | 2 | impact | Pins the database version. Different versions carry different measured curves, so this changes predictions even with every other input fixed. |
| 11 | `DYN_AIC_MTP_SEED` | 1 | 0 | 0 | no impact | RNG seed for mocker MTP burst sampling. Governs reproducibility between runs, not expected performance — which is exactly why it should be pinned when comparing two runs. |

**Summary:** 10 `impact`, 1 `no impact`.

---

## 9. Frontend — Multimodal (1)

One entry; the rest of the multimodal surface is shared and appears in category 20.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_MULTIMODAL_LOADER_CACHE_GB` | 2 | 3 | 6 | impact | Size of the frontend's media-loader cache for fetched images and video. A miss means re-fetching the asset over the network and re-decoding it, both on the request path, so on repeated-media traffic this sets a large fraction of TTFT. |

**Summary:** 1 `impact`, 0 `no impact`.

---

## 10. Frontend — CLI-only flags (5)

The frontend's five flags with no environment variable. Three are SGLang-native pass-throughs accepted only under `--dyn-chat-processor sglang`; the effect that earns them a tag is that specifying a parser turns parsing on, and specifying a template changes what gets rendered into the prompt.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `--chat-template` | 2 | 3 | 6 | impact | Replaces the model's chat template. A different template renders a different prompt, so prompt token count, prefill cost, and prefix-cache alignment all move — the largest effect available from a flag that looks like formatting. |
| 2 | `--tool-call-parser` | 2 | 2 | 4 | impact | Selects the tool-call parser for the SGLang processor. Naming one enables incremental tool-call parsing on the streaming path, which is work that does not happen otherwise. |
| 3 | `--reasoning-parser` | 2 | 2 | 4 | impact | Same for reasoning parsing: unspecified means no reasoning parsing is performed at all. |
| 4 | `--router-kv-overlap-score-weight` | 1 | 2 | 2 | impact | Deprecated and hidden, still read and still perturbs the routing cost function. Matches the verdict on its two environment variables in category 1. |
| 5 | `--admission-control` | 0 | 0 | 0 | no impact | Deprecated and discarded — accepted so old launch commands keep starting, sets nothing. |

**Summary:** 4 `impact`, 1 `no impact`.

---

## 11. Shared — Request / event plane (19)

The transport under every request and every KV event. Six entries are bind addresses and ports; the rest are the dials on the hot path itself. Two host settings are tagged for a reason worth stating: on a multi-NIC host, auto-detection can land the KV/response path on a management interface instead of the fabric, which is a hundredfold throughput difference dressed up as an address.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_REQUEST_PLANE_CODEC` | 3 | 3 | 9 | impact | Payload codec for every request and response — `json` is the compatibility default, `msgpack` the faster one. Serialisation is paid twice per request on the frontend and twice on the worker. |
| 2 | `DYN_EVENT_PLANE` | 2 | 3 | 6 | impact | `zmq` by default, `nats` optionally. The transport carrying KV events; its throughput and latency set how fresh the router's cache view is. |
| 3 | `DYN_TCP_POOL_SIZE` | 2 | 3 | 6 | impact | Connections per peer, default 50. Below the concurrency the frontend actually drives, requests serialise behind the pool instead of the worker. |
| 4 | `DYN_TCP_REQUEST_TIMEOUT` | 2 | 3 | 6 | impact | Request timeout, default 10 s. Long generations and slow prefills can exceed it; when they do the failure looks like a worker problem rather than a timeout. |
| 5 | `DYN_TCP_RESPONSE_STREAM_HOST` | 2 | 3 | 6 | impact | Interface for the response-stream server. Unset it auto-detects a routable IP, which on a multi-NIC node may not be the fast fabric. Tagged for interface selection, not for the string. |
| 6 | `DYN_TCP_RPC_HOST` | 2 | 3 | 6 | impact | Advertised bind host for the request-plane listener, with the same auto-detection caveat and the same consequence. |
| 7 | `DYN_EVENT_PLANE_CODEC` | 2 | 2 | 4 | impact | `json` or `msgpack` for events. KV event volume scales with block churn, so codec cost is paid continuously on both publisher and subscriber. |
| 8 | `DYN_TCP_CHANNEL_BUFFER` | 2 | 2 | 4 | impact | Per-stream channel depth, default 100. Sets where backpressure lands between the transport and the consumer. |
| 9 | `DYN_TCP_LATENCY_TRACE` | 2 | 2 | 4 | impact | Per-hop latency traces for request-plane calls. Instrumentation on the transport path — switched on precisely when measuring, which is when it costs. |
| 10 | `DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY` | 1 | 2 | 2 | impact | Subscriber channel capacity. Too small and KV events are dropped under burst, which degrades router accuracy silently rather than loudly. |
| 11 | `DYN_ZMQ_BROKER_ENABLED` | 1 | 2 | 2 | impact | Routes events through a broker instead of direct connections: one more hop on every event, in exchange for connection scalability. |
| 12 | `DYN_TCP_CONNECT_TIMEOUT` | 1 | 2 | 2 | impact | Connect timeout, default 3 s. Bounds how long a stuck peer stalls a new connection; off the steady-state path. |
| 13 | `DYN_TCP_MAX_MESSAGE_SIZE` | 1 | 2 | 2 | impact | Ceiling on a single request-plane message. Large multimodal or long-context payloads are what approach it. |
| 14 | `DYN_TCP_RPC_PORT` | 0 | 0 | 0 | no impact | Fixed port instead of OS-assigned. |
| 15 | `DYN_TCP_RESPONSE_STREAM_PORT` | 0 | 0 | 0 | no impact | Port for the response-stream server; 0 or unset means ephemeral. |
| 16 | `DYN_ZMQ_BROKER_URL` | 0 | 0 | 0 | no impact | Explicit broker address, overriding discovery. |
| 17 | `ZMQ_BROKER_XSUB_BIND` | 0 | 0 | 0 | no impact | XSUB bind address, broker binary only. |
| 18 | `ZMQ_BROKER_XPUB_BIND` | 0 | 0 | 0 | no impact | XPUB bind address, broker binary only. |
| 19 | `ZMQ_BROKER_NAMESPACE` | 0 | 0 | 0 | no impact | Namespace for broker discovery registration. |

**Summary:** 13 `impact`, 6 `no impact`.

---

## 12. Shared — KVBM (KV block manager) (20)

vLLM and TensorRT-LLM only. KVBM is the multi-tier KV cache — GPU to CPU to disk to object storage — so the tier sizes here decide how much of a prefix survives eviction, which is the same currency the KV router trades in. The addressing and credential entries are inert; two location settings are tagged on the same locality reasoning as the NIC selection in category 11.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_RUNTIME_ENABLED_KVBM` | 3 | 3 | 9 | impact | The gate that turns the KVBM connector on inside the engine process. Everything else in this category is dead without it, and enabling it changes the engine's cache hierarchy outright. |
| 2 | `DYN_KVBM_CPU_CACHE_GB` | 3 | 3 | 9 | impact | Size of the CPU offload tier. The dominant term in whether an evicted prefix can be restored instead of recomputed, and the one the router's host-cache-hit weight is pricing. |
| 3 | `DYN_KVBM_DISK_CACHE_GB` | 3 | 3 | 9 | impact | Size of the disk tier, same reasoning one level down. Restores are slower, so the tier is worth less per byte but far larger. |
| 4 | `DYN_KVBM_TRANSFER_BATCH_SIZE` | 2 | 3 | 6 | impact | Blocks per transfer batch. Sets offload and restore throughput, and therefore whether the lower tiers can keep up with eviction pressure at all. |
| 5 | `DYN_KVBM_OBJECT_ENABLED` | 2 | 3 | 6 | impact | Adds an object-storage tier below disk. A new, much slower tier in the restore path — worthwhile only when the prefix reuse horizon is long. |
| 6 | `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | 2 | 2 | 4 | impact | Disables the filter that decides which blocks are worth writing to disk. Without it, disk bandwidth is spent on blocks that will never be read back. |
| 7 | `DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR` | 2 | 2 | 4 | impact | Merges engine and KVBM events into one stream. Changes both event volume on the wire and the accuracy of the router's view of the lower tiers. |
| 8 | `DYN_KVBM_OBJECT_NUM_BLOCKS` | 2 | 2 | 4 | impact | Capacity of the object tier in blocks; the same size-versus-hit-rate trade as the tiers above. |
| 9 | `DYN_KVBM_ENABLE_RECORD` | 2 | 2 | 4 | impact | Records KVBM activity for debugging. Diagnostic instrumentation in the transfer path. |
| 10 | `DYN_KVBM_OBJECT_REGION` | 1 | 3 | 3 | impact | Region of the object store. Cross-region restores are an order of magnitude slower than same-region ones, and nothing about the configuration will look wrong. |
| 11 | `DYN_KVBM_OBJECT_ENDPOINT` | 1 | 3 | 3 | impact | Object-store endpoint. A local MinIO and a remote S3 are the same setting and wildly different restore latencies. |
| 12 | `DYN_FATBIN_PATH` | 1 | 2 | 2 | impact | Custom CUDA fatbin for the block transfer kernels. Substitutes the code doing the copying. |
| 13 | `DYN_KVBM_METRICS` | 1 | 1 | 1 | no impact | Exposes the KVBM metrics endpoint. Scrape-time cost only. |
| 14 | `DYN_KVBM_METRICS_PORT` | 0 | 0 | 0 | no impact | Port for that endpoint. |
| 15 | `DYN_KVBM_OBJECT_BUCKET` | 0 | 0 | 0 | no impact | Bucket name, with a `{worker_id}` template for per-worker buckets. |
| 16 | `DYN_KVBM_OBJECT_ACCESS_KEY` | 0 | 0 | 0 | no impact | Object-store credential. |
| 17 | `DYN_KVBM_OBJECT_SECRET_KEY` | 0 | 0 | 0 | no impact | Object-store credential. |
| 18 | `DYN_KVBM_LEADER_ZMQ_HOST` | 0 | 0 | 0 | no impact | Leader ZMQ host; intra-node control plane. |
| 19 | `DYN_KVBM_LEADER_ZMQ_PUB_PORT` | 0 | 0 | 0 | no impact | Leader ZMQ publish port. |
| 20 | `DYN_KVBM_LEADER_ZMQ_ACK_PORT` | 0 | 0 | 0 | no impact | Leader ZMQ acknowledgement port. |

**Summary:** 12 `impact`, 8 `no impact`.

---

## 13. Shared — Worker runtime & identity (21)

The workers' common arg group. It mixes pure identity — endpoint strings, instance ids — with several settings that change how much work the engine is asked to do per request. The template and parser entries deserve particular attention: they alter the rendered prompt or add per-chunk work, and they are easy to read as formatting choices.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_ENGINE_REQUEST_LIMIT` | 3 | 3 | 9 | impact | Worker-pool semaphore size, disabled by default. Setting it caps engine concurrency and enables worker-side rejection, so it bounds throughput deliberately and converts overload into refusals. |
| 2 | `DYN_CONNECTOR` | 2 | 3 | 6 | impact | Selects the KV transfer connector for TensorRT-LLM — nixl, lmcache, kvbm, null. A different implementation of the disaggregated KV path, not a parameter of one. |
| 3 | `DYN_DEFAULT_THINKING_MODE` | 2 | 3 | 6 | impact | Deployment-wide default for chat-template thinking. Enabling thinking multiplies output length on models that support it, which moves every decode-side number at once. |
| 4 | `DYN_CUSTOM_JINJA_TEMPLATE` | 2 | 3 | 6 | impact | Overrides the model's chat template. Different rendering means a different prompt token count and different prefix alignment; the worker-side twin of `--chat-template` in category 10. |
| 5 | `DYN_MULTIMODAL_EMBEDDING_CACHE_CAPACITY_GB` | 2 | 3 | 6 | impact | Embedding cache size, 0 and therefore disabled by default. On repeated-media traffic a hit skips vision encoding entirely. |
| 6 | `DYN_ENABLE_STRUCTURAL_TAG` | 2 | 2 | 4 | impact | Structural-tag guided decoding for tool calls. Constrained decoding costs per token; it buys well-formed output with sampling-time work. |
| 7 | `DYN_STRUCTURAL_TAG_SCOPE` | 2 | 2 | 4 | impact | `auto` or `always`. Decides how much traffic pays that constrained-decoding cost. |
| 8 | `DYN_STRUCTURAL_TAG_SCHEMA` | 2 | 2 | 4 | impact | `auto` or `strict`. Full parameter schemas produce larger grammars, and grammar size is what constrained decoding pays for. |
| 9 | `DYN_TOOL_CALL_PARSER` | 2 | 2 | 4 | impact | Names the tool-call parser. Unset means no tool-call parsing happens; naming one adds parsing to every response that might contain a call. |
| 10 | `DYN_REASONING_PARSER` | 2 | 2 | 4 | impact | Same for reasoning: unspecified means no reasoning parsing is performed. |
| 11 | `DYN_MULTIMODAL_EMBEDDING_CACHE_PUBLISHER` | 2 | 2 | 4 | impact | Publishes embedding-cache state so the KV router can route on it. Adds event traffic; pays off only with KV-aware routing. |
| 12 | `DYN_MEDIA_OUTPUT_FS_URL` | 1 | 3 | 3 | impact | Where generated images and video are written. `file://` on local disk and `s3://` are the same setting with very different write latency on the response path. |
| 13 | `PYTHONHASHSEED` | 1 | 3 | 3 | impact | Set to 0 by the workers before engine start so prefix hashes agree across processes. Left random, block hashing diverges between processes and prefix reuse silently degrades. |
| 14 | `DYN_HEALTH_CHECK_PAYLOAD` | 1 | 2 | 2 | impact | Overrides the health canary payload. The canary is a real inference request on a timer — a heavy payload puts real periodic load on the worker under test. |
| 15 | `DYN_ENDPOINT` | 0 | 0 | 0 | no impact | The `dyn://namespace.component.endpoint` this worker serves. Identity. |
| 16 | `DYN_KV_STATE_ENDPOINT` | 0 | 0 | 0 | no impact | Which endpoint owns this worker's KV event and recovery state. Identity. |
| 17 | `DYN_ENDPOINT_TYPES` | 0 | 0 | 0 | no impact | Which endpoint types to register — chat, completions, none. Surface, not cost. |
| 18 | `DYN_STABLE_ROUTING_ID` | 0 | 0 | 0 | no impact | Pins the routing identity across restarts. An identity choice; any effect on router state continuity is a consequence of restarting, not of serving. |
| 19 | `DYN_SELF_HOST_METADATA` | 0 | 0 | 0 | no impact | Serve the model card from this worker instead of publishing it through discovery. Startup-path only. |
| 20 | `DYN_MEDIA_OUTPUT_HTTP_URL` | 0 | 0 | 0 | no impact | Rewrites media paths in responses into URLs. String substitution. |
| 21 | `DYN_OUTPUT_MODALITIES` | 0 | 0 | 0 | no impact | Declares which modalities this worker produces. Defines the subject, per the category 5 convention. |

**Summary:** 14 `impact`, 7 `no impact`.

---

## 14. Shared — Request tracing (24)

Per-request tracing, plus ten deprecated `DYN_AUDIT_*` aliases that still work and therefore inherit their target's verdict. The thing to know: the master switch is cheap, but `DYN_REQUEST_TRACE_RECORDS=request_payload` serialises whole request bodies on the request path. That is the setting that turns tracing from an accounting cost into a throughput cost.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_REQUEST_TRACE_RECORDS` | 2 | 3 | 6 | impact | Selects which records are emitted. `request_end` is a small structured record; `request_payload` serialises the entire request body per request. Same switch, different order of magnitude. |
| 2 | `DYN_REQUEST_TRACE` | 2 | 3 | 6 | impact | Master switch. Off, none of this exists; on, every request produces at least one record through the in-process bus. |
| 3 | `DYN_REQUEST_TRACE_SINKS` | 2 | 3 | 6 | impact | Where records go — `file`, `stderr`, `nats`, `otel`, `s3`. A remote sink puts network work behind the request path; `stderr` puts it behind a lock. |
| 4 | `DYN_REQUEST_TRACE_CAPACITY` | 2 | 2 | 4 | impact | In-process bus capacity. Too small and producers block or records drop under exactly the load worth tracing. |
| 5 | `DYN_REQUEST_TRACE_FILE_FORMAT` | 2 | 2 | 4 | impact | `jsonl` or `jsonl_gz`. Compression moves the cost from disk bandwidth to CPU on the writer. |
| 6 | `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | 2 | 2 | 4 | impact | Write buffer size; sets how often the sink touches the filesystem. |
| 7 | `DYN_AUDIT_SINKS` | 1 | 3 | 3 | impact | Deprecated alias for the above, still read. Legacy `jsonl`/`jsonl_gz` values map onto the file sink. |
| 8 | `DYN_AUDIT_FORCE_LOGGING` | 1 | 3 | 3 | impact | Deprecated shim for `DYN_REQUEST_TRACE_RECORDS=request_payload` — it turns on the expensive mode by a name that does not say so. |
| 9 | `DYN_AUDIT_CAPACITY` | 1 | 2 | 2 | impact | Deprecated alias for the trace bus capacity. |
| 10 | `DYN_AUDIT_JSONL_BUFFER_BYTES` | 1 | 2 | 2 | impact | Deprecated alias for the file sink buffer size. |
| 11 | `DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS` | 1 | 2 | 2 | impact | Deprecated alias for the file sink flush interval. |
| 12 | `DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES` | 1 | 2 | 2 | impact | Deprecated alias capping OTLP payload size; the cap is what stops large bodies dominating export. |
| 13 | `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | 1 | 2 | 2 | impact | Rotation threshold in uncompressed bytes. Rotation is a burst of close, compress, and open work. |
| 14 | `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | 1 | 2 | 2 | impact | Rotation threshold in records, same reasoning. |
| 15 | `DYN_AUDIT_JSONL_GZ_ROLL_BYTES` | 1 | 2 | 2 | impact | Deprecated alias for the byte roll threshold. |
| 16 | `DYN_AUDIT_JSONL_GZ_ROLL_LINES` | 1 | 2 | 2 | impact | Deprecated alias for the line roll threshold. |
| 17 | `DYN_REQUEST_TRACE_S3_REGION` | 1 | 2 | 2 | impact | Region for the S3 sink. Cross-region uploads are slower and the sink is fed from the request path. |
| 18 | `DYN_REQUEST_TRACE_FILE_PATH` | 0 | 0 | 0 | no impact | Output path on local disk. |
| 19 | `DYN_REQUEST_TRACE_OUTPUT_PATH` | 0 | 0 | 0 | no impact | Deprecated alias for that path. |
| 20 | `DYN_REQUEST_TRACE_NATS_SUBJECT` | 0 | 0 | 0 | no impact | Subject the NATS sink publishes to. |
| 21 | `DYN_REQUEST_TRACE_S3_BUCKET` | 0 | 0 | 0 | no impact | Bucket for the S3 sink. |
| 22 | `DYN_REQUEST_TRACE_S3_PREFIX` | 0 | 0 | 0 | no impact | Key prefix within that bucket. |
| 23 | `DYN_AUDIT_NATS_SUBJECT` | 0 | 0 | 0 | no impact | Deprecated alias for the NATS subject. |
| 24 | `DYN_AUDIT_OUTPUT_PATH` | 0 | 0 | 0 | no impact | Deprecated alias for the file path. |

**Summary:** 17 `impact`, 7 `no impact`.

---

## 15. Shared — Tokio runtime (4)

Four settings, all of them real. This is the thread pool every Rust component runs on, so it is as close to the machine as this catalogue gets.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_RUNTIME_NUM_WORKER_THREADS` | 3 | 3 | 9 | impact | Async worker threads. Sets the frontend's parallelism ceiling directly, and interacts with CPU pinning — the first thing to fix before comparing two frontend runs. |
| 2 | `DYN_ENABLE_POLL_HISTOGRAM` | 2 | 3 | 6 | impact | Tokio task poll-time histogram. The documentation states it adds roughly 2x overhead — a diagnostic that materially changes what it measures. |
| 3 | `DYN_RUNTIME_MAX_BLOCKING_THREADS` | 2 | 3 | 6 | impact | Ceiling on the blocking pool. Blocking work — filesystem, some tokenizer paths — queues behind it when it is too small. |
| 4 | `DYN_RUNTIME_INHIBITED_DURATION_SECS` | 2 | 2 | 4 | impact | How long a worker is excluded locally after a request failure. Zero disables it. Directly shapes recovery behaviour after a transient failure, and therefore the tail during one. |

**Summary:** 4 `impact`, 0 `no impact`.

---

## 16. Shared — Forward-pass metric trace (6)

FPM tracing on the workers. Row 1 does two things at once, which is the part worth knowing.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_FPM_TRACE` | 2 | 3 | 6 | impact | Persists forward-pass metrics to rotating gzip JSONL — and, per the help text, also enables the backend FPM instrumentation needed to produce them. Two costs behind one switch: instrumentation in the engine loop and compression on the writer. |
| 2 | `DYN_FPM_MODE` | 2 | 2 | 4 | impact | `sampled` keeps the latest event per DP rank per interval; `full` keeps every event reaching the tap. The difference is bounded work against unbounded. |
| 3 | `DYN_FPM_SAMPLE_INTERVAL_MS` | 2 | 2 | 4 | impact | Sampling interval under `sampled`. Sets the rate of everything downstream of the tap. |
| 4 | `DYN_FPM_JSONL_GZ_ROLL_BYTES` | 1 | 2 | 2 | impact | Rotation threshold; rotation bursts compression and file work. |
| 5 | `DYN_FPM_MAX_SEGMENTS` | 0 | 0 | 0 | no impact | Retention — how many segments are kept per producer. Disk footprint, not throughput. |
| 6 | `DYN_FPM_OUTPUT_PATH` | 0 | 0 | 0 | no impact | Segment path prefix. |

**Summary:** 4 `impact`, 2 `no impact`.

---

## 17. Shared — Health checks (7)

Mostly timeouts on a supervision path. The one that matters is the canary, because it is not a probe in the usual sense — it is a real inference request issued on a timer against the worker under test.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_HEALTH_CHECK_ENABLED` | 2 | 2 | 4 | impact | Turns on the periodic endpoint canary. That canary occupies an engine slot each time it fires, so it is measurable load, not just liveness. |
| 2 | `DYN_ENGINE_HEALTH_CHECK_INTERVAL` | 1 | 2 | 2 | impact | How often the engine monitor probes. Frequency sets how much of the above is paid. |
| 3 | `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | 1 | 1 | 1 | no impact | Timeout for one canary request. Bounds a failure, does not add load. |
| 4 | `DYN_ENGINE_HEALTH_CHECK_TIMEOUT` | 1 | 1 | 1 | no impact | Timeout for one engine liveness probe. |
| 5 | `DYN_ENGINE_HEALTH_SHUTDOWN_TIMEOUT` | 1 | 1 | 1 | no impact | Grace after an unhealthy engine before the worker exits. Affects how fast a bad worker leaves the fleet, not how a good one serves. |
| 6 | `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` | 0 | 0 | 0 | no impact | Which endpoints gate the system health response. Reporting. |
| 7 | `DYN_CANARY_WAIT_TIME` | 0 | 0 | 0 | no impact | Wait time for canary deployments. Rollout timing. |

**Summary:** 2 `impact`, 5 `no impact`.

---

## 18. Shared — Topology & KV transfer (5)

Constrains disaggregated KV transfer to a topology domain, so that prefill and decode workers exchanging KV are on the same fabric. Off by default; when on, it is a routing constraint with real teeth.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_TOPOLOGY_ENABLED` | 2 | 3 | 6 | impact | Reads the topology domain files. Off, KV transfer pairs are chosen without regard to fabric locality — which on a multi-rack deployment is the difference between NVLink and the network. |
| 2 | `DYN_KV_TRANSFER_ENFORCEMENT` | 2 | 3 | 6 | impact | `strict` rejects cross-domain pairings outright, `preferred` penalises them. Strict trades placement flexibility for guaranteed locality and can leave capacity unusable. |
| 3 | `DYN_KV_TRANSFER_DOMAIN` | 2 | 2 | 4 | impact | The domain this worker advertises. Mislabel it and the constraint enforces the wrong thing with full confidence. |
| 4 | `DYN_KV_TRANSFER_PREFERRED_WEIGHT` | 2 | 2 | 4 | impact | Penalty applied to cross-domain candidates under `preferred`. The dial between locality and spread. |
| 5 | `DYN_TOPOLOGY_MOUNT_PATH` | 0 | 0 | 0 | no impact | Where the domain files are mounted. |

**Summary:** 4 `impact`, 1 `no impact`.

---

## 19. Shared — Multimodal HTTP fetch client (7)

The worker-side client that fetches media referenced by a request. Every one of these is on the request path for multimodal traffic, because the fetch happens before the engine sees the prompt — a slow or starved fetch client shows up as TTFT and looks like engine latency.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_HTTP_CONCURRENCY` | 2 | 3 | 6 | impact | Process-wide cap on in-flight fetches, default 50, httpx only. Backpressure in front of the pool: too low and media requests queue in the worker before any inference starts. |
| 2 | `DYN_HTTP_MAX_CONNECTIONS` | 2 | 3 | 6 | impact | Total pool size, default 100. The ceiling on parallel media fetches, and therefore on multimodal throughput when assets are remote. |
| 3 | `DYN_HTTP_TIMEOUT` | 2 | 2 | 4 | impact | Per-call override replacing the caller's timeout on every fetch. Sets how long a slow asset can hold a request open. |
| 4 | `DYN_HTTP_CONNECT_TIMEOUT` | 2 | 2 | 4 | impact | Handshake budget, default 5 s, deliberately independent so a stuck origin fails fast rather than consuming the whole call budget. |
| 5 | `DYN_HTTP_POOL_TIMEOUT` | 2 | 2 | 4 | impact | Wait-for-free-slot timeout, default 60 s, httpx only. What happens once the two ceilings above are hit. |
| 6 | `DYN_HTTP_MAX_KEEPALIVE` | 2 | 2 | 4 | impact | Idle keepalive cap, 0 meaning match the pool size. Too low and every fetch pays a fresh handshake. |
| 7 | `DYN_HTTP_KEEPALIVE_TIMEOUT` | 2 | 2 | 4 | impact | How long an idle connection stays warm, default 15 s, aiohttp only. Same handshake-amortisation trade. |

**Summary:** 7 `impact`, 0 `no impact`.

---

## 20. Shared — Multimodal (4)

Two decoding parameters and two security gates.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_MM_VIDEO_NUM_FRAMES` | 3 | 3 | 9 | impact | Frames sampled per video, default 32. Vision encoding cost is close to linear in this, and the sampled frames become tokens — it moves both encode time and prompt length. |
| 2 | `DYN_MM_IMAGE_CACHE_SIZE` | 2 | 3 | 6 | impact | Decoded-image LRU, default 8 entries. That is small; on repeated-image traffic the difference between hitting and missing is a full fetch and decode. |
| 3 | `DYN_MM_LOCAL_PATH` | 1 | 1 | 1 | no impact | Allowlists a directory for `file://` media. A gate on whether such inputs are accepted at all. |
| 4 | `DYN_MM_ALLOW_INTERNAL` | 0 | 0 | 0 | no impact | Permits media URLs resolving to private addresses. SSRF protection; off by default for good reason. |

**Summary:** 2 `impact`, 2 `no impact`.

---

## 21. Shared — Model download (10)

Everything here runs before the process serves its first request. Cold start is a performance number that matters — it is what ModelExpress exists to shorten and what the Planner pays for on every scale-up — but it is not steady-state serving, so the three tags below are cold-start tags and nothing in this category can move a TTFT.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `MODEL_EXPRESS_NO_SHARED_STORAGE` | 1 | 3 | 3 | impact | Streams model files over gRPC instead of relying on shared storage. A different weight-loading path with a different cold-start time — the whole point of the setting. |
| 2 | `HF_HUB_OFFLINE` | 1 | 2 | 2 | impact | Skips Hub API calls when the model is already cached, removing network round-trips from startup. Also removes a failure mode where a slow Hub delays every replica. |
| 3 | `HF_ENDPOINT` | 1 | 2 | 2 | impact | Points at a Hub mirror. A local mirror and the public Hub are the same setting with very different download throughput. |
| 4 | `HF_HUB_CACHE` | 0 | 0 | 0 | no impact | Hub cache directory. |
| 5 | `HF_HOME` | 0 | 0 | 0 | no impact | Hugging Face home directory. |
| 6 | `HF_TOKEN` | 0 | 0 | 0 | no impact | Hub authentication token. |
| 7 | `HUGGING_FACE_HUB_TOKEN` | 0 | 0 | 0 | no impact | Deprecated alias for that token. |
| 8 | `HF_TOKEN_PATH` | 0 | 0 | 0 | no impact | Path to the stored token. |
| 9 | `MODEL_EXPRESS_CACHE_PATH` | 0 | 0 | 0 | no impact | ModelExpress cache path. |
| 10 | `MODEL_EXPRESS_URL` | 0 | 0 | 0 | no impact | Deprecated and inert for vLLM — the plugin reads its own configuration. |

**Summary:** 3 `impact`, 7 `no impact`.

---

## 22. Shared — Logging (6)

Log level is the classic accidental throughput regression: debug logging on a per-request path costs formatting, allocation, and I/O for every line, on every request, in every process. The rest are formatting choices.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_LOG` | 2 | 3 | 6 | impact | Log level. `debug` or `trace` on a serving path is a large, easily-overlooked regression — and it is exactly what gets left on after an investigation. |
| 2 | `DYN_LOGGING_SPAN_EVENTS` | 2 | 2 | 4 | impact | Emits span create and close events. Doubles line volume on instrumented paths for a modest gain in traceability. |
| 3 | `DYN_LOGGING_JSONL` | 1 | 2 | 2 | impact | Structured JSONL output. Serialising a JSON object per line costs more than formatting a text line, and it is normal to leave it on in production. |
| 4 | `DYN_LOGGING_CONFIG_PATH` | 0 | 0 | 0 | no impact | Path to a logging configuration file. What it contains may matter; the path does not. |
| 5 | `DYN_LOG_USE_LOCAL_TZ` | 0 | 0 | 0 | no impact | Local timezone instead of UTC in timestamps. |
| 6 | `DYN_SDK_DISABLE_ANSI_LOGGING` | 0 | 0 | 0 | no impact | Disables ANSI colour codes. |

**Summary:** 3 `impact`, 3 `no impact`.

---

## 23. Shared — OpenTelemetry export (9)

Distributed tracing. The switch and the sample ratio are the cost; the endpoints are addresses. Export is batched and off the critical path, which is why the endpoint entries stay untagged even though a remote collector is further away than a local one.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `OTEL_EXPORT_ENABLED` | 2 | 3 | 6 | impact | Turns on OTLP export for traces and logs. Span creation, attribute recording, and batching are paid per request once this is on. |
| 2 | `OTEL_TRACES_SAMPLE_RATIO` | 2 | 3 | 6 | impact | Fraction of traces sampled. The single dial that scales the cost above — `0.01` and `1.0` are two very different overheads from one string. |
| 3 | `OTEL_EXPORTER_OTLP_PROTOCOL` | 1 | 2 | 2 | impact | `grpc` or `http/protobuf`. Different serialisation and connection behaviour on the export path. |
| 4 | `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | 1 | 2 | 2 | impact | Per-signal override of that protocol for traces. |
| 5 | `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` | 1 | 2 | 2 | impact | Per-signal override for logs. |
| 6 | `OTEL_EXPORTER_OTLP_ENDPOINT` | 0 | 0 | 0 | no impact | Generic collector endpoint. |
| 7 | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | 0 | 0 | 0 | no impact | Collector endpoint for traces. |
| 8 | `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | 0 | 0 | 0 | no impact | Collector endpoint for logs. |
| 9 | `OTEL_SERVICE_NAME` | 0 | 0 | 0 | no impact | Service name attached to exported signals. |

**Summary:** 5 `impact`, 4 `no impact`.

---

## 24. Shared — NATS (8)

Connection and authentication for NATS. Only relevant to the data path when the request or event plane is set to `nats`; the five auth entries never are.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_NATS_REQUEST_TIMEOUT_SECS` | 2 | 2 | 4 | impact | Request/reply timeout, defaulting to the async-nats 10 s. When NATS carries the request plane, this is the ceiling on a single request — long generations can exceed it. |
| 2 | `DYN_NATS_STREAM_MAX_AGE` | 1 | 2 | 2 | impact | Retention age for stream messages. Sets how much history the server holds, and therefore its memory. |
| 3 | `NATS_SERVER` | 0 | 0 | 0 | no impact | Server address. |
| 4 | `NATS_AUTH_USERNAME` | 0 | 0 | 0 | no impact | Credential. |
| 5 | `NATS_AUTH_PASSWORD` | 0 | 0 | 0 | no impact | Credential. |
| 6 | `NATS_AUTH_TOKEN` | 0 | 0 | 0 | no impact | Credential. |
| 7 | `NATS_AUTH_NKEY` | 0 | 0 | 0 | no impact | Credential. |
| 8 | `NATS_AUTH_CREDENTIALS_FILE` | 0 | 0 | 0 | no impact | Credential file. |

**Summary:** 2 `impact`, 6 `no impact`.

---

## 25. Shared — etcd (7)

Discovery storage. Off the request path entirely; the one tagged entry is tagged because it decides how quickly a dead worker stops receiving traffic.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `ETCD_LEASE_TTL` | 2 | 2 | 4 | impact | Lease TTL, default 10 s. A registration outlives its worker by up to this long, and during that window the router can still choose it — a tail-latency and error-rate effect during any worker loss, and a scale-down artefact under autoscaling. |
| 2 | `ETCD_ENDPOINTS` | 0 | 0 | 0 | no impact | Comma-separated etcd URLs. |
| 3 | `ETCD_AUTH_USERNAME` | 0 | 0 | 0 | no impact | Credential. |
| 4 | `ETCD_AUTH_PASSWORD` | 0 | 0 | 0 | no impact | Credential. |
| 5 | `ETCD_AUTH_CA` | 0 | 0 | 0 | no impact | CA certificate for etcd TLS. |
| 6 | `ETCD_AUTH_CLIENT_CERT` | 0 | 0 | 0 | no impact | Client certificate for etcd TLS. |
| 7 | `ETCD_AUTH_CLIENT_KEY` | 0 | 0 | 0 | no impact | Client key for etcd TLS. |

**Summary:** 1 `impact`, 6 `no impact`.

---

## 26. Shared — System status server (6)

The health and metrics sidecar server. A clean sweep: it serves probes and a metrics endpoint, and none of its configuration touches the inference path. The scrape cost belongs to the scrape, not to these settings.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_SYSTEM_ENABLED` | 1 | 1 | 1 | no impact | Enables the status server. Marked deprecated and slated for removal; an idle probe server is not a measurable cost. |
| 2 | `DYN_SYSTEM_PORT` | 0 | 0 | 0 | no impact | Status server port. |
| 3 | `DYN_SYSTEM_HOST` | 0 | 0 | 0 | no impact | Status server host. |
| 4 | `DYN_SYSTEM_HEALTH_PATH` | 0 | 0 | 0 | no impact | Health endpoint path. |
| 5 | `DYN_SYSTEM_LIVE_PATH` | 0 | 0 | 0 | no impact | Liveness endpoint path. |
| 6 | `DYN_SYSTEM_STARTING_HEALTH_STATUS` | 0 | 0 | 0 | no impact | Health status reported during startup. |

**Summary:** 0 `impact`, 6 `no impact`.

---

## 27. Shared — Shutdown & lifecycle (3)

All three govern teardown. They cost wall-clock between runs — which is worth knowing when a paired A/B restarts components dozens of times — but nothing measured during a run depends on them. This follows the same reasoning as `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` in category 4.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_PREFILL_DRAIN_TIMEOUT_S` | 1 | 1 | 1 | no impact | Budget for draining in-flight prefill during shutdown, default 30 s. Bounds how long teardown waits. |
| 2 | `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` | 1 | 1 | 1 | no impact | Overall graceful shutdown timeout for a worker. |
| 3 | `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` | 1 | 1 | 1 | no impact | Delay before shutdown begins so load balancers can drain first. |

**Summary:** 0 `impact`, 3 `no impact`.

---

## 28. Shared — Discovery (2)

How components find each other.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_DISCOVERY_BACKEND` | 2 | 3 | 6 | impact | `kubernetes`, `etcd`, `file`, or `mem`. These have very different watch latency and failure behaviour — `mem` and `file` remove a network dependency entirely, which is why they are the sane choice for a single-node frontend benchmark and why comparing across backends is not comparing like with like. |
| 2 | `DYN_KUBE_DISCOVERY_MODE` | 1 | 2 | 2 | impact | `pod` or `container` registration granularity. Container mode registers each container independently, multiplying the instance count the router tracks and the discovery events it processes. |

**Summary:** 2 `impact`, 0 `no impact`.

---

## 29. Shared — Profiling (2)

NVTX range emission, one switch per language runtime. Both exist to be turned on during a profiling session, and both cost something while on — the reason to tag them is that a profiling run with them enabled is not measuring the same system as a run without.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_ENABLE_RUST_NVTX` | 2 | 2 | 4 | impact | NVTX ranges from the Rust runtime for Nsight Systems. Range push and pop on instrumented paths. |
| 2 | `DYN_NVTX` | 2 | 2 | 4 | impact | NVTX ranges from the Python layer, where the per-range cost is higher. |

**Summary:** 2 `impact`, 0 `no impact`.

---

## 30. Shared — Memory (1)

One entry, and a large one.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_MEMORY_DISABLE_NUMA` | 2 | 3 | 6 | impact | Disables NUMA-aware host allocation. On a multi-socket node that means host KV buffers can land on the far socket from the GPU that reads them, adding cross-socket traffic to every offload and restore. On a single-socket node it changes nothing — which is what makes it easy to dismiss. |

**Summary:** 1 `impact`, 0 `no impact`.

---

## 31. Shared — remaining small groups (9)

The tail of the shared set: four entries the page files under Frontend core, plus LoRA, RL, the Mooncake endpoint, and `--version`. `DYN_REQUEST_PLANE` is here for filing reasons and is one of the most consequential settings in the whole catalogue.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_REQUEST_PLANE` | 3 | 3 | 9 | impact | `tcp` or `nats` for every request from router to worker. The help text says plainly that tcp is fastest. This is the transport under the entire serving path, and switching it changes latency, throughput, and failure behaviour together. |
| 2 | `DYN_LORA_ENABLED` | 2 | 3 | 6 | impact | Turns on LoRA adapter support. Adapter-aware handling per request, and it is the precondition for the whole of category 7. |
| 3 | `DYN_ENABLE_RL` | 2 | 2 | 4 | impact | Enables RL training support. Beyond mounting the RL router on the frontend, it selects RL-friendly vLLM defaults for TITO and per-token logprob parity — decode-path behaviour changes, not just an endpoint. |
| 4 | `DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE` | 2 | 2 | 4 | impact | On by default: drops tool definitions from the rendered template when `tool_choice='none'`. Tool schemas are large, so this removes prompt tokens from every such request. |
| 5 | `DYN_MOONCAKE_KV_EVENTS_ENDPOINT` | 1 | 3 | 3 | impact | The Mooncake/HiCache master the router queries for shared-cache hits, and that SGLang workers register against. Paired with `--shared-cache-type hicache` in category 1; wrong endpoint means the lookup fails on the routing path. |
| 6 | `DYN_NAMESPACE` | 0 | 0 | 0 | no impact | Namespace for model discovery scoping. Identity. |
| 7 | `DYN_LORA_PATH` | 0 | 0 | 0 | no impact | LoRA cache directory. |
| 8 | `DYN_DUMP_CONFIG_TO` | 0 | 0 | 0 | no impact | Writes resolved configuration to a file at startup. |
| 9 | `--version` | 0 | 0 | 0 | no impact | Prints the version and exits. |

**Summary:** 5 `impact`, 4 `no impact`.

---

## 32. vLLM — vLLM engine wrapper (21)

Dynamo's own wrapper around vLLM, not vLLM's arguments — those arrive as pass-through and are the engine's business. Eleven of the 21 configure the optional startup self-benchmark, which is unusual in this catalogue: those settings do not change how the worker serves, they change how long it takes to become ready and how precisely the resulting performance model describes the engine. They are tagged on that basis, and separated below.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_VLLM_DISAGGREGATION_MODE` | 3 | 3 | 9 | impact | `agg`, `pd`, `prefill`, `decode`, or `encode`. Decides whether this worker does prefill, decode, both, or vision encoding — the single largest structural choice available to a worker, and the one the whole disaggregation story rests on. |
| 2 | `DYN_VLLM_USE_TOKENIZER` | 3 | 3 | 9 | impact | Bypasses Dynamo's preprocessor in favour of vLLM's tokenizer, leaving only `/v1/chat/completions` available through the frontend. Moves tokenization from the Rust frontend into the Python worker — it relocates the cost rather than removing it. |
| 3 | `DYN_VLLM_ROUTE_TO_ENCODER` | 2 | 3 | 6 | impact | Routes multimodal work to separate encoder workers. Adds a network hop and an embedding transfer per request, in exchange for encoding in parallel with decode on other GPUs. |
| 4 | `DYN_VLLM_EMBEDDING_TRANSFER_MODE` | 2 | 3 | 6 | impact | `local` via the filesystem, or `nixl-write`/`nixl-read` over RDMA. Three genuinely different transports for embeddings on the multimodal request path. |
| 5 | `DYN_VLLM_GMS_SHADOW_MODE` | 2 | 3 | 6 | impact | Shadow engines skip KV cache allocation, pause after init, and resume when the active engine dies. Changes both memory footprint and failover latency; requires `--load-format=gms`. |
| 6 | `DYN_BENCHMARK_MODE` | 2 | 3 | 6 | impact | Runs a self-benchmark sweep before accepting requests. Readiness is delayed by minutes and the GPU is fully occupied while it runs — on an autoscaled fleet that is a real cost. |
| 7 | `DYN_VLLM_HEADLESS` | 2 | 2 | 4 | impact | Secondary nodes run vLLM workers with no Dynamo endpoints, for multi-node TP/PP. Determines the multi-node topology this worker participates in. |
| 8 | `DYN_VLLM_ENABLE_MULTIMODAL` | 2 | 2 | 4 | impact | Capability gate for multimodal processing; without it none of the multimodal components can be used, so it precedes several rows here and in category 34. |
| 9 | `DYN_VLLM_MM_PROMPT_TEMPLATE` | 2 | 2 | 4 | impact | Constructs the final multimodal prompt around `<prompt>` and the media placeholder. Prompt construction is prompt length. |
| 10 | `DYN_CUSTOM_ENCODER_CLASS` | 2 | 2 | 4 | impact | Substitutes a custom `VisionEncoderBackend` run in-process by the aggregated worker. The encoder is the multimodal hot path; this replaces it wholesale. |
| 11 | `DYN_VLLM_EMBEDDING_WORKER` | 2 | 2 | 4 | impact | Runs as a pooling/embedding worker, which skips KV events, KV router registration, and scheduler instrumentation — none of which apply to pooling models. |
| 12 | `DYN_BENCHMARK_WARMUP_ITERATIONS` | 2 | 2 | 4 | impact | Warmup iterations before the self-benchmark, default 5. Too few and the sweep measures a cold engine, which is the classic way to produce a wrong performance model. |
| 13 | `DYN_BENCHMARK_POINTS_FILE` | 2 | 2 | 4 | impact | Explicit benchmark points replacing generated grid sampling for the selected phases. Redefines what the sweep measures. |
| 14 | `DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES` | 1 | 2 | 2 | impact | Caps prefill new-token samples, default 64. Fewer points means a shorter sweep and a coarser prefill curve. |
| 15 | `DYN_PREFILL_MAX_KV_READ_TOKEN_SAMPLES` | 1 | 2 | 2 | impact | Caps prefill KV-read samples per point, default 16. Same startup-time against resolution trade. |
| 16 | `DYN_DECODE_MAX_KV_READ_TOKEN_SAMPLES` | 1 | 2 | 2 | impact | Caps decode KV-read samples per batch size, default 128. |
| 17 | `DYN_DECODE_MAX_BATCH_SIZE_SAMPLES` | 1 | 2 | 2 | impact | Caps decode batch-size samples, default 128, always retaining the minimum and feasible maximum. |
| 18 | `DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES` | 1 | 2 | 2 | impact | Caps prefill request-batch-size samples per new-token point, default 3. |
| 19 | `DYN_BENCHMARK_TIMEOUT` | 1 | 2 | 2 | impact | Soft limit on the sweep, default 900 s, after which partial results are returned and startup continues. Bounds the readiness delay and can silently truncate the model. |
| 20 | `DYN_VLLM_REALTIME` | 0 | 0 | 0 | no impact | Serves a bidirectional realtime endpoint over `/v1/realtime`. A different protocol and workload rather than a tuning choice. |
| 21 | `DYN_BENCHMARK_OUTPUT_PATH` | 0 | 0 | 0 | no impact | Where the results JSON is written. |

**Summary:** 19 `impact`, 2 `no impact`.

---

## 33. vLLM — vLLM extras (non-CLI) (8)

Environment variables the vLLM worker reads or sets directly, several of them vLLM's own. Dynamo sets some of these on the engine's behalf, which is worth knowing before overriding one.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_SPLIT_ENCODE` | 2 | 3 | 6 | impact | Defaults to 1: splits encode work out of the prefill worker on the multimodal prefill path. Changes which process does vision encoding and therefore what competes with prefill for the GPU. |
| 2 | `VLLM_NIXL_SIDE_CHANNEL_HOST` | 2 | 3 | 6 | impact | The host vLLM's NIXL connector advertises for its KV side channel. Dynamo derives it from the worker's resolved IP; overriding it to the wrong interface puts disaggregated KV transfer on the wrong network. |
| 3 | `VLLM_WORKER_MULTIPROC_METHOD` | 2 | 2 | 4 | impact | Multiprocessing start method — fork against spawn changes worker startup time and initial memory sharing. |
| 4 | `DYN_GMS_SCRATCH_KV_ENABLED` | 2 | 2 | 4 | impact | Applies GPU Memory Service scratch-KV patches, set automatically in headless mode. Alters KV allocation inside the engine. |
| 5 | `VLLM_LOG_STATS_INTERVAL` | 1 | 2 | 2 | impact | How often vLLM logs engine stats; Dynamo aligns it with its own metric polling. Frequent stats logging is work in the engine loop. |
| 6 | `PROMETHEUS_MULTIPROC_DIR` | 0 | 0 | 0 | no impact | Directory for multiprocess Prometheus collectors when vLLM runs several engine processes. |
| 7 | `VLLM_CONFIGURE_LOGGING` | 0 | 0 | 0 | no impact | Hands logging configuration to Dynamo instead of vLLM. |
| 8 | `VLLM_NO_USAGE_STATS` | 0 | 0 | 0 | no impact | Disables vLLM usage telemetry — a startup-time call, not a serving cost. |

**Summary:** 5 `impact`, 3 `no impact`.

---

## 34. vLLM — LoRA and Multimodal (4)

The vLLM-specific tail.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_VLLM_FRONTEND_DECODING` | 2 | 3 | 6 | impact | Images are decoded in the Rust frontend and moved to the backend over NIXL RDMA, bypassing in-engine decode. Moves decode work off the GPU node and puts an RDMA transfer in its place — a substantial relocation on image-heavy traffic. |
| 2 | `DYN_LORA_HOTSWAP_ENABLED` | 2 | 2 | 4 | impact | Permits adapters to be swapped on a running engine. Swapping costs load time on the serving path; the alternative is not serving that adapter at all. |
| 3 | `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | 1 | 2 | 2 | impact | vLLM's own prerequisite for runtime LoRA updates, set by the Dynamo worker when LoRA is enabled. Gates the behaviour above. |
| 4 | `VLLM_LORA_MODULES_LOADING_TIMEOUT` | 1 | 1 | 1 | no impact | How long vLLM waits when loading LoRA modules. Bounds a failure. |

**Summary:** 3 `impact`, 1 `no impact`.

---

## 35. TensorRT-LLM — engine wrapper (49)

The largest category in the catalogue, and the one with the highest density of genuine engine knobs — unlike vLLM and SGLang, TensorRT-LLM settings that Dynamo does not wrap arrive through the `--extra-engine-args` YAML rather than as pass-through flags, so more of the real surface is here. Roughly half the rows are diffusion and DiT settings that are inert for LLM serving but decisive for image and video generation; they are ordered below the LLM knobs rather than dismissed.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_TRTLLM_TENSOR_PARALLEL_SIZE` | 3 | 3 | 9 | impact | Tensor parallelism. Sets how many GPUs serve one model instance, and therefore per-request latency, aggregate throughput, and how much KV cache each instance holds. |
| 2 | `DYN_TRTLLM_FREE_GPU_MEMORY_FRACTION` | 3 | 3 | 9 | impact | Fraction of free GPU memory given to KV cache after weights and buffers, default 0.9. Directly sets KV capacity, which sets concurrency and prefix-reuse headroom — the highest-leverage single number in the category. |
| 3 | `DYN_TRTLLM_MAX_NUM_TOKENS` | 3 | 3 | 9 | impact | Batched input tokens per iteration. The engine's prefill batch budget, and the denominator the router's queue and busy thresholds are expressed against. |
| 4 | `DYN_TRTLLM_MAX_BATCH_SIZE` | 3 | 3 | 9 | impact | Maximum requests the engine will schedule together. The decode batch ceiling, and therefore the throughput-latency trade. |
| 5 | `DYN_TRTLLM_DISAGGREGATION_MODE` | 3 | 3 | 9 | impact | `agg`, `pd`, `prefill`, `decode`, or `encode`. The worker's structural role, as in vLLM. |
| 6 | `DYN_TRTLLM_EXTRA_ENGINE_ARGS` | 3 | 3 | 9 | impact | YAML of arbitrary additional engine arguments. This is the escape hatch through which most of TensorRT-LLM's real tuning surface arrives, so its contents can change anything. |
| 7 | `DYN_TRTLLM_OVERRIDE_ENGINE_ARGS` | 3 | 3 | 9 | impact | Dictionary string overriding entries from that YAML — the documented example toggles `kv_cache_config.enable_block_reuse`, which is prefix caching itself. |
| 8 | `DYN_TRTLLM_ENABLE_ATTENTION_DP` | 2 | 3 | 6 | impact | Attention data parallelism, with `attention_dp_size` equal to tensor-parallel size. A different parallelism strategy for attention, with its own latency and balance characteristics. |
| 9 | `DYN_TRTLLM_PIPELINE_PARALLEL_SIZE` | 2 | 3 | 6 | impact | Pipeline parallelism. Trades bubble overhead against fitting larger models; interacts with batch size for pipeline fill. |
| 10 | `DYN_TRTLLM_EXPERT_PARALLEL_SIZE` | 2 | 3 | 6 | impact | Expert parallelism for MoE models — the placement of experts across GPUs, and the all-to-all traffic that follows. |
| 11 | `DYN_TRTLLM_MAX_SEQ_LEN` | 2 | 3 | 6 | impact | Maximum total request length. Caps the workload and shapes memory reservation; deduced from the model config when unset. |
| 12 | `DYN_TRTLLM_KV_BLOCK_SIZE` | 2 | 3 | 6 | impact | KV block size, default 32. Sets reuse granularity in the engine, and must agree with the router's `DYN_KV_CACHE_BLOCK_SIZE` for prefix matching to work. |
| 13 | `DYN_TRTLLM_ENABLE_CUDA_GRAPH` | 2 | 3 | 6 | impact | CUDA graph capture for the transformer forward pass. Removes launch overhead from decode, which is where launch overhead dominates. Mutually exclusive with torch.compile. |
| 14 | `DYN_TRTLLM_QUANT_ALGO` | 2 | 3 | 6 | impact | Quantization algorithm — FP8, NVFP4, AWQ, and so on. Changes arithmetic throughput and memory footprint together, and with them everything downstream. |
| 15 | `DYN_TRTLLM_TORCH_DTYPE` | 2 | 3 | 6 | impact | Model dtype. `float32` against `bfloat16` is roughly a factor of two in memory and bandwidth before any other consideration. |
| 16 | `DYN_TRTLLM_ATTN_BACKEND` | 2 | 3 | 6 | impact | `VANILLA` PyTorch SDPA against `TRTLLM` kernels for diffusion models. A kernel-level substitution in the hottest loop. |
| 17 | `DYN_TRTLLM_ENABLE_TEACACHE` | 2 | 3 | 6 | impact | TeaCache skips redundant diffusion steps. Named for speed, and it is one of the larger wins available in the diffusion path. |
| 18 | `DYN_TRTLLM_TEACACHE_THRESH` | 2 | 3 | 6 | impact | TeaCache threshold, default 0.2 — how aggressively steps are skipped, trading fidelity for speed. |
| 19 | `DYN_TRTLLM_DEFAULT_NUM_INFERENCE_STEPS` | 2 | 3 | 6 | impact | Default diffusion steps, default 50. Generation time is close to linear in step count — the dominant cost of an image or video request. |
| 20 | `DYN_TRTLLM_DEFAULT_NUM_FRAMES` | 2 | 3 | 6 | impact | Default frames per video, default 81. The other linear term in video generation cost. |
| 21 | `DYN_TRTLLM_DEFAULT_HEIGHT` | 2 | 3 | 6 | impact | Default output height, default 480. Diffusion cost scales with pixel count, so this and width multiply. |
| 22 | `DYN_TRTLLM_DEFAULT_WIDTH` | 2 | 3 | 6 | impact | Default output width, default 832. Same reasoning. |
| 23 | `DYN_TRTLLM_LOAD_FORMAT` | 2 | 2 | 4 | impact | Weight loading format, `auto` or `gms`. Cold-start path, and the prerequisite for GPU Memory Service sharing. |
| 24 | `DYN_TRTLLM_DISABLE_TORCH_COMPILE` | 2 | 2 | 4 | impact | Turns off torch.compile. Compilation costs startup time and buys steady-state speed; disabling it inverts that trade. |
| 25 | `DYN_TRTLLM_ENABLE_FULLGRAPH` | 2 | 2 | 4 | impact | torch.compile fullgraph mode — stricter capture, potentially faster, and it fails rather than falling back. |
| 26 | `DYN_TRTLLM_GUIDED_DECODING_BACKEND` | 2 | 2 | 4 | impact | `xgrammar` or `llguidance` for structured output. Different grammar engines with different per-token costs. |
| 27 | `DYN_TRTLLM_MAX_BEAM_WIDTH` | 2 | 2 | 4 | impact | Beam search width. Multiplies decode work per request when above 1. |
| 28 | `DYN_ENGINE_CONV_AFFINITY` | 2 | 2 | 4 | impact | Forces conversation-affinity ADP routing regardless of engine detection. Pins a conversation to a DP rank, trading balance for cache locality across turns. |
| 29 | `DYN_ENGINE_CONV_AFFINITY_DP_RANK_SOURCE` | 2 | 2 | 4 | impact | `engine` lets TensorRT-LLM load-balance the first request; `dynamo` forwards the router's chosen rank. Decides who owns initial attention-DP placement. |
| 30 | `DYN_TRTLLM_PUBLISH_KV_EVENTS` | 2 | 2 | 4 | impact | Publishes KV cache events to the router. Without them the KV router falls back to predicted state — the worker-side half of category 1 row 1. |
| 31 | `DYN_TRTLLM_SKIP_WARMUP` | 2 | 2 | 4 | impact | Skips warmup inference at init. Faster readiness, at the cost of the first real requests hitting an unwarmed engine — which is measured unless warmup is handled by the harness. |
| 32 | `DYN_TRTLLM_GPUS_PER_NODE` | 2 | 2 | 4 | impact | GPUs per node, inferred when unset. Wrong values distort placement across a multi-node deployment. |
| 33 | `DYN_TRTLLM_ENABLE_LAYERWISE_NVTX_MARKER` | 2 | 2 | 4 | impact | Per-layer NVTX markers. Fine-grained instrumentation inside the forward pass — high resolution, and high enough overhead to distort what it profiles. |
| 34 | `DYN_TRTLLM_QUANT_DYNAMIC` | 2 | 2 | 4 | impact | On-the-fly quantization of BF16 weights during loading, on by default. Startup cost against not needing pre-quantized checkpoints. |
| 35 | `DYN_TRTLLM_ENABLE_MULTIMODAL` | 2 | 2 | 4 | impact | Enables multimodal request processing — the vision path and its cost. |
| 36 | `DYN_TRTLLM_MODALITY` | 2 | 2 | 4 | impact | Model modality, retained mainly for diffusion. Selects an entirely different generation pipeline. |
| 37 | `DYN_TRTLLM_TEACACHE_USE_RET_STEPS` | 2 | 2 | 4 | impact | Retention steps for TeaCache; adjusts which steps are protected from skipping. |
| 38 | `DYN_TRTLLM_DEFAULT_NUM_IMAGES_PER_PROMPT` | 2 | 2 | 4 | impact | Images generated per prompt. A direct multiplier on work per request. |
| 39 | `DYN_TRTLLM_DIT_ULYSSES_SIZE` | 2 | 2 | 4 | impact | Ulysses sequence parallelism for DiT. Distributes diffusion work across GPUs, with its own communication cost. |
| 40 | `DYN_TRTLLM_DIT_RING_SIZE` | 2 | 2 | 4 | impact | Ring parallelism for DiT, same trade with a different communication pattern. |
| 41 | `DYN_TRTLLM_DIT_CFG_SIZE` | 2 | 2 | 4 | impact | CFG parallelism for DiT — splits classifier-free guidance branches across devices. |
| 42 | `DYN_TRTLLM_MODEL_LOADER_EXTRA_CONFIG` | 1 | 2 | 2 | impact | Extra loader configuration such as `gms_read_only`. Cold-start behaviour and memory sharing. |
| 43 | `DYN_TRTLLM_DEFAULT_GUIDANCE_SCALE` | 1 | 2 | 2 | impact | Default CFG scale, default 5.0. Affects output quality and, where CFG is batched, the work per step. |
| 44 | `DYN_TRTLLM_MAX_FILE_SIZE_MB` | 1 | 1 | 1 | no impact | Ceiling on downloadable embedding files and image URLs, default 50. An acceptance gate. |
| 45 | `DYN_TRTLLM_ALLOWED_LOCAL_MEDIA_PATH` | 0 | 0 | 0 | no impact | Directory the model may read local media from. A security allowlist. |
| 46 | `DYN_TRTLLM_ENCODE_ENDPOINT` | 0 | 0 | 0 | no impact | Address of the encode worker. Identity. |
| 47 | `DYN_TRTLLM_MODEL` | 0 | 0 | 0 | no impact | Model path or Hub identifier. Defines the subject under test. |
| 48 | `DYN_TRTLLM_SERVED_MODEL_NAME` | 0 | 0 | 0 | no impact | Name the model is served under. |
| 49 | `DYN_TRTLLM_REVISION` | 0 | 0 | 0 | no impact | Hub revision for download. Subject definition; a different revision is a different model, not a different setting. |

**Summary:** 43 `impact`, 6 `no impact`.

---

## 36. TensorRT-LLM — extras, KVBM, Multimodal (5)

The TensorRT-LLM tail. Row 1 is unusually direct about what it is for.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_TRTLLM_SERVER_DISABLE_GC` | 2 | 3 | 6 | impact | Disables Python cyclic GC in the worker to remove GC pauses from the hot path. Its entire purpose is a latency artefact — GC pauses show up in the tail, and this is the switch that removes them. |
| 2 | `DYN_TRTLLM_FRONTEND_DECODING` | 2 | 3 | 6 | impact | Images decoded in the Rust frontend and moved over NIXL RDMA instead of decoded in-engine. Relocates vision decode off the GPU node. |
| 3 | `DYN_ENABLE_TEST_LOGITS_PROCESSOR` | 1 | 3 | 3 | impact | Installs a dummy logits processor. Described as a test-only smoke hook, but a logits processor runs per token per request — leaving it on in a measured run is a real and easily-missed cost. |
| 4 | `DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS` | 1 | 2 | 2 | impact | Deprecated alias for `--publish-kv-events`, still read and still effective; same verdict as its replacement in category 35. |
| 5 | `DYN_KVBM_TRTLLM_ZMQ_PORT` | 0 | 0 | 0 | no impact | Port for the KVBM event consolidator's ZMQ channel. |

**Summary:** 4 `impact`, 1 `no impact`.

---

## 37. SGLang — engine wrapper (12)

Dynamo's SGLang wrapper is the thinnest of the three — SGLang's own `ServerArgs` carry most of the tuning, and arrive as pass-through. What remains is topology and role selection, plus one trace level that is easy to leave too high.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_SGL_USE_TOKENIZER` | 3 | 3 | 9 | impact | Deprecated but live: routes pre/post processing through SGLang's tokenizer instead of Dynamo's preprocessor, moving tokenization from the Rust frontend into the Python worker. The replacement is `--dyn-chat-processor sglang` on the frontend, which makes the same choice explicitly. |
| 2 | `SGLANG_TRACE_LEVEL` | 2 | 3 | 6 | impact | Trace verbosity 1 to 4 when tracing is enabled, default 2 (per-request). Level 3 adds the decode loop and level 4 is full — instrumentation inside the generation loop, where per-step overhead compounds. |
| 3 | `DYN_SGL_DISAGG_CONFIG` | 2 | 3 | 6 | impact | YAML disaggregation configuration. Defines the prefill/decode split and the transfer arrangement between them. |
| 4 | `DYN_SGL_EMBEDDING_TRANSFER_MODE` | 2 | 3 | 6 | impact | `local`, `nixl-write`, or `nixl-read` for embeddings — filesystem against RDMA on the multimodal path. |
| 5 | `DYN_SGL_DEDICATED_MM_ENCODER` | 2 | 3 | 6 | impact | Selects the internal topology with a dedicated multimodal encode worker, required on PD/P/D workers that consume precomputed embeddings. A topology choice, and getting it wrong misroutes embeddings. |
| 6 | `DYN_SGL_DISAGG_CONFIG_KEY` | 2 | 2 | 4 | impact | Which key of a nested disaggregation config this worker adopts — `prefill` or `decode`. Selects the role, so a wrong key gives a correctly-configured worker in the wrong job. |
| 7 | `DYN_SGL_ENABLE_MULTIMODAL` | 2 | 2 | 4 | impact | Capability gate for raw multimodal inputs, and the precondition for the vision path's cost. |
| 8 | `DYN_SGL_EMBEDDING_WORKER` | 2 | 2 | 4 | impact | Runs as an embedding worker and sets SGLang's `--is-embedding`. A different workload and a different engine configuration. |
| 9 | `DYN_SGL_ENABLE_RL` | 2 | 2 | 4 | impact | Enables RL metadata upload support. Adds upload work alongside serving. |
| 10 | `DYN_SGL_IMAGE_DIFFUSION_WORKER` | 0 | 0 | 0 | no impact | Runs as an image diffusion worker. Selects what the worker is, not how fast it is. |
| 11 | `DYN_SGL_VIDEO_GENERATION_WORKER` | 0 | 0 | 0 | no impact | Runs as a video generation worker. Same reasoning. |
| 12 | `DYN_SGLANG_ENGINE_ROUTES` | 0 | 0 | 0 | no impact | Exposes trusted SGLang methods under `/engine/<path>`. Additional routes; cost belongs to whatever is called. |

**Summary:** 9 `impact`, 3 `no impact`.

---

## 38. SGLang — extras and Multimodal (5)

The SGLang tail.

| # | Setting | L | M | Score | Proposed | Why |
|---|---------|---|---|------:|----------|-----|
| 1 | `DYN_SGL_FRONTEND_DECODING` | 2 | 3 | 6 | impact | Frontend image decode with NIXL RDMA transfer instead of in-engine decode, as in the other two backends. |
| 2 | `DYN_SGL_ALLOW_TOP_LOGPROBS` | 1 | 3 | 3 | impact | Overrides the guard blocking top-logprobs on SGLang. Beyond the reliability warning the guard exists for, computing top-k logprobs per token is real decode-side work. |
| 3 | `DYN_SKIP_SGLANG_LOG_FORMATTING` | 1 | 2 | 2 | impact | Leaves SGLang's own log formatting alone instead of reformatting every line into the Dynamo format. Reformatting is per-line work on whatever SGLang logs. |
| 4 | `DYN_FORWARDPASS_METRIC_PORT` | 0 | 0 | 0 | no impact | ZMQ port SGLang publishes forward-pass metrics on, used instead of the default system-port wiring. |
| 5 | `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | 0 | 0 | 0 | no impact | SGLang's switch for blocking non-zero-rank child processes, set by the Dynamo worker. |

**Summary:** 3 `impact`, 2 `no impact`.

---

## Result

All 411 settings have a proposed verdict: **277 `impact`, 134 `no impact`.** None of them is applied — the catalogue page still reports every setting as `unexamined`, and will until these are validated. Proposals cover the full set with no gaps and no duplicates, checked against the catalogue's own record list.

| Group | Settings | impact | no impact |
|-------|---------:|-------:|----------:|
| Frontend | 127 | 88 | 39 |
| Shared | 180 | 103 | 77 |
| vLLM | 33 | 27 | 6 |
| TensorRT-LLM | 54 | 47 | 7 |
| SGLang | 17 | 12 | 5 |

The two-thirds `impact` share is high, and that is a property of the catalogue rather than a soft hand: settings that only name things — ports, paths, credentials, endpoints, metric prefixes — are a minority of what Dynamo exposes, and they are concentrated in a few categories (system status server, NATS and etcd auth, the route-path overrides) that came out almost entirely `no impact`.

### Conventions applied, in the order they were settled

1. **Deprecated is judged on behaviour, not status** (category 2). Still read and still effective keeps its verdict; accepted and discarded is `no impact`.
2. **Subject versus setting** (category 5). A setting that silently changes a workload the operator asked for is `impact`; one that names the subject under test — model path, revision, served name, worker role — is `no impact`.
3. **Measurement fidelity counts** (categories 2, 6). Settings that change the number you read rather than the system you read it from are tagged: startup gates that let a run begin against a partial fleet, and histogram buckets that determine reported quantiles.
4. **The observer effect counts** (categories 5, 11, 15, 29, 35). Instrumentation that is switched on precisely when someone is measuring — `DYN_DEBUG_PERF`, `DYN_ENABLE_POLL_HISTOGRAM` at a documented ~2x, NVTX ranges, layerwise markers, latency traces — is tagged.
5. **Locality is a setting, addressing is not** (categories 11, 12, 14, 21, 33). Something that selects which physical resource is used — a network interface, an object-store region, a Hub mirror — is `impact`; a name or path within one resource is not.
6. **Teardown is not measured** (categories 4, 27). Shutdown and drain timeouts cost wall-clock between runs and change nothing during one.

### Where to push back first

- **Category 6, the six histogram bucket configs.** Tagged on convention 3. If the tag should mean the system's own speed, these six flip together and are the cleanest group to reverse.
- **Category 4 rows 7 and 8.** `DYN_HTTP_OVERLOAD_STATUS_CODE` and `DYN_DISABLE_FRONTEND_NVEXT` are tagged on a client's retry behaviour and a harness's routing overrides — effects real to a measurement, external to the process.
- **Category 8, the AIC inputs.** Ten `impact` verdicts on settings that are inert unless an experimental load model or the mocker is running. Tagged on the cost of being wrong, not on how often they are set.
- **Category 21, model download.** Three cold-start tags in a category that cannot move a serving metric. Defensible if cold start counts as performance, which it does for the Planner and autoscaling; not if the tag is about steady-state serving.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Knob Catalog

Use this file for Dynamo-owned deployment, routing, transport, cache, and scaling controls. Engine-native controls
belong in the [vLLM](vllm.md), [SGLang](sglang.md), and [TensorRT-LLM](tensorrt-llm.md) guides.

This catalog does not rank levers or prescribe an order. Each row names a related knob family, not a bundle to apply
together. Change one independently testable knob per candidate under the
[one-variable rule](../../rules/optimization/one-variable.md); use a coupled change only when functionality requires it
or evidence supports the interaction.

DGD paths below use the v1beta1 `spec.components[*]` layout. For a v1alpha1 manifest, resolve the corresponding
component under `spec.services.<name>` and use the schema supported by the installed operator. Verify every selected
flag and default against the running Dynamo version, then prove from the rendered pod and startup logs that it engaged.
Model identity, runtime image, secret references, ports, namespaces, and working directories are deployment wiring or
fixed target constraints, not optimization knobs, and are intentionally omitted.

## Deployment Shape and Placement

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| The serving architecture does not match the workload | aggregated versus disaggregated component graph; `spec.components[*].type` (`worker`, `prefill`, `decode`) | changes prefill/decode isolation, KV-transfer cost, TTFT, ITL, and throughput | rendered DGD, registered worker roles, KV-transfer logs, AIPerf | a topology conversion is a functional multi-field bundle and starts a new deployment comparison |
| Total serving capacity is too low or too high | worker, prefill, or decode `spec.components[*].replicas` | changes sustainable concurrency, queueing, latency, throughput, and GPU use | ready replica count, active GPU count, request distribution, AIPerf | changing replicas changes the resource budget; do not attribute the result to per-GPU efficiency |
| Prefill and decode capacity are imbalanced | prefill and decode `replicas` | reduces starvation or idle capacity and can improve end-to-end throughput and latency | role-level utilization or queue evidence when available, complete AIPerf result | a coordinated count change is one topology hypothesis only when required to preserve a fixed GPU budget; see [rate matching](../rate-matching/matching.md) |
| A model instance spans the wrong number of GPUs or nodes | container GPU requests and `spec.components[*].multinode.nodeCount` | changes model fit, communication overhead, replica count, and capacity | scheduled nodes and GPUs, backend parallel shape, communication logs, AIPerf | GPU requests, node count, and engine parallelism must agree; consult [parallelism](../model-sizing/parallelism.md) |
| CPU or memory throttling limits a component | `spec.components[*].podTemplate.spec.containers[name=main].resources` | can reduce frontend, router, tokenizer, or transfer stalls | Kubernetes throttling, CPU, memory, OOM, and component latency metrics | a larger resource request may change node placement; distinguish resource relief from placement effects |
| Workers land on unsuitable or distant hardware | `nodeSelector`, node affinity, tolerations, and DGD `topologyConstraint.packDomain` | can reduce interconnect and cross-node latency or ensure the intended GPU class | pod-to-node mapping, GPU type, topology labels, NCCL/NIXL transport, AIPerf | placement changes may also change node quality and invalidate node equivalence |
| Cross-domain prefill-to-decode KV transfers are expensive | `spec.experimental.kvTransferPolicy` topology source, `domain`, `enforcement`, and `preferredWeight` | can reduce KV-transfer latency and congestion | selected prefill/decode domains, transfer path, fallback count, AIPerf | choose exactly one topology source: `clusterTopologyName` or `labelKey`; `preferredWeight` applies only to `preferred`, and `required` can strand usable capacity |
| Disaggregated KV transfer falls back to a slow path | worker RDMA resource requests, `IPC_LOCK`, `UCX_TLS`, `UCX_RNDV_SCHEME`, `UCX_RNDV_THRESH`, and the backend's NIXL/UCX transfer configuration | can engage the intended fast fabric and reduce KV-transfer time, TTFT, and stalls | RDMA devices and pod resources, NIXL/UCX logs, absence of TCP fallback, transfer bandwidth, AIPerf | these fields may form a functionality-required engagement bundle; exact resources and transports are cluster- and backend-specific |
| Shared-memory pressure or IPC failure limits workers | `spec.components[*].sharedMemorySize` | prevents `/dev/shm` exhaustion and may restore expected IPC behavior | mounted tmpfs size, worker logs, OOM or IPC failures | capacity beyond the workload requirement is not a serving optimization |
| Repeated compilation dominates rollout time | `spec.components[*].compilationCache.{pvcName,mountPath}` | reduces later pod startup and warmup time | cache mount, cache reuse logs, ready time | startup-only; do not claim a steady-state AIPerf improvement |

## Frontend, Runtime, and Transport

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Frontend saturation limits the endpoint | frontend `replicas` and frontend CPU or memory resources | raises request-processing capacity and can reduce endpoint queueing | per-pod CPU, event-loop or request latency, load distribution, AIPerf | multiple frontend replicas require routing-state behavior appropriate for the selected router mode |
| Tokenization or chat processing is CPU-bound | `--dyn-chat-processor`; `--dyn-preprocess-workers`; `--tokenizer` | can lower frontend processing time and raise request throughput | frontend timing logs, CPU use, token counts, AIPerf | processor choices are experimental; preprocess workers apply only to the vLLM or SGLang processor, and `fastokens` affects BPE encoding only |
| Request distribution overhead is visible | `--request-plane` / `DYN_REQUEST_PLANE` (`tcp`, `nats`) | changes router-to-worker latency and throughput | resolved configuration, transport logs, frontend overhead, AIPerf | changing the plane changes infrastructure and failure behavior, not just performance |
| TCP setup, buffering, or payload size is limiting | `DYN_TCP_POOL_SIZE`, `DYN_TCP_CHANNEL_BUFFER`, `DYN_TCP_MAX_MESSAGE_SIZE`, `DYN_TCP_REQUEST_TIMEOUT`, `DYN_TCP_CONNECT_TIMEOUT` | can reduce connection churn, backpressure, oversized-message failures, or timeout loss | TCP metrics and errors, connection reuse, payload sizes, AIPerf | advanced diagnostics; larger pools and buffers consume memory, while timeouts do not increase engine capacity |
| KV event transport is delayed or overloaded | `--event-plane` / `DYN_EVENT_PLANE` (`zmq`, `nats`) | changes cache-state propagation latency and event overhead | resolved plane, event lag or loss, router state, AIPerf | affects KV-aware features; a plane change also changes infrastructure dependencies |
| Worker concurrency exceeds the engine's safe capacity | `--engine-request-limit` / `DYN_ENGINE_REQUEST_LIMIT` | bounds active engine requests and can trade overload failures for controlled rejection | accepted, queued, rejected, and active request counts; errors; AIPerf | a limit below usable engine capacity leaves throughput idle; unset disables worker-side rejection |
| Short bursts overflow after worker admission is enabled | `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` | absorbs a bounded burst before rejection | Dynamo queue depth, rejection count, latency distribution | advanced and effective only with `DYN_ENGINE_REQUEST_LIMIT`; a deeper queue can hide overload as latency |
| Router block accounting disagrees with the backend | frontend `--kv-cache-block-size` / `DYN_KV_CACHE_BLOCK_SIZE`; standalone router `--router-block-size` | restores correct cache-overlap and capacity accounting | backend block size, router model card or startup log, cache-hit behavior | this is a consistency setting, not a free tuning value; all participants must use the same block size |
| Worker loss should migrate eligible in-flight requests | `--migration-limit`; `--migration-max-seq-len` | may reduce failed requests during worker disconnects | injected or observed worker loss, migration count, errors, memory use | resilience path; state tracking consumes memory and does not improve healthy steady-state serving |
| Router starts before enough workers are present | `--router-min-initial-workers` | avoids early requests hitting an incomplete pool | startup logs, discovered and ready worker count | startup-only and can delay readiness indefinitely if set above attainable capacity |
| Discovery churn delays worker visibility | `--discovery-backend` / `DYN_DISCOVERY_BACKEND` | changes component discovery latency and control-plane load | discovery logs, worker registration delay, request errors | usually not a request hot-path optimization and requires the corresponding infrastructure |

## Request Routing

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| The current routing policy creates skew or misses reuse | `--router-mode` / `DYN_ROUTER_MODE`: `round-robin`, `random`, `power-of-two`, `least-loaded`, `kv`, `direct`, or `device-aware-weighted` | changes load spread, cache locality, TTFT, ITL, and throughput | per-worker request and load distribution, cache evidence for `kv`, AIPerf | `power-of-two` and `least-loaded` use the synchronous prefill path in disaggregated prefill mode; `direct` requires a compatible topology |
| Cache reuse is unhelpful and load should dominate KV routing | `--load-aware` / `DYN_ROUTER_LOAD_AWARE` | reduces hot-worker concentration when prefix reuse is absent | active-block and prefill-token load, request spread, AIPerf | this preset changes several router fields together; treat it as one documented mode, not as evidence for each field |
| KV routing over- or under-values device-local prefixes | `--router-kv-overlap-score-credit` | shifts the tradeoff between prefix reuse and worker load | device-local overlap, cache-hit rate, per-worker load, AIPerf | high credit can keep routing to a loaded worker |
| A loaded worker receives too much overlap credit | `--router-kv-overlap-score-credit-decay`; `--router-prefill-load-scale` | reduces load skew while retaining some locality benefit | overlap-adjusted scores, active prefill load, request spread, AIPerf | these are separate knobs; change only the one supported by the observed scoring problem |
| CPU- or disk-tier reuse is valued incorrectly | `--router-host-cache-hit-weight`; `--router-disk-cache-hit-weight` | changes routing toward workers holding lower-tier KV | tier-specific hits, transfer time, worker load, AIPerf | useful only when the corresponding KVBM tiers and router state are engaged |
| Deterministic KV routing causes persistent hotspots | `--router-temperature` | may spread requests across similarly scored workers and improve tails | score distribution, worker skew, cache-hit rate, AIPerf | more randomness can reduce locality; zero is deterministic |
| Router cache state is inaccurate or event handling is costly | `--router-kv-events` / `--no-router-kv-events`; `--router-ttl-secs` | trades worker-reported cache state for approximate prediction | event receipt or predicted entries, state age, cache hits, AIPerf | TTL applies only when KV events are disabled; disabling events can route on stale predictions |
| KV events arrive after the next related request | `--router-predicted-ttl-secs` | bridges short event delay with predict-on-route state | side-indexer entries, event delay, cache-hit rate, AIPerf | requires KV events and is independent of approximate-mode `--router-ttl-secs` |
| Router load estimates omit important active work | `--router-track-active-blocks`; `--router-track-output-blocks`; `--router-track-prefill-tokens`; `--router-assume-kv-reuse` | changes estimated worker load and routing balance | router active-block and prefill-token metrics, worker skew, AIPerf | each field is independent; output tracking and reuse assumptions must match workload behavior |
| Static prompt-load accounting remains stale too long | `--router-prefill-load-model` (`none`, `aic`); required AIC identity: `--aic-backend`, `--aic-system`, and `--aic-model-path` | can improve prompt-load decay and worker selection | AIC engagement, predicted duration, active prefill load, AIPerf | experimental; `aic` requires KV mode, prefill-token tracking, Dynamo chat processing, and model identity matching the deployed engine |
| All workers are overloaded and requests should wait centrally | `--router-queue-threshold`; `--router-queue-policy`; `--router-policy-config` | trades rejection or worker queueing for controlled router queueing and changes average or tail TTFT | router queue depth and wait, policy selection, errors, AIPerf | threshold enables queueing; a policy-config file is a functional configuration surface, and deeper waits do not add capacity |
| Queued requests dispatch using stale prefix-overlap scores | `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS` | refreshes cache-locality information after a long queue wait and can improve final worker selection | queue wait, overlap-refresh logs, selected worker, cache hits, TTFT, AIPerf | active only with router queueing and overlap refresh support; lower values add index work, while `0` disables refresh |
| KV index maintenance is router-CPU-bound | `--router-event-threads`; `--use-remote-indexer`; `--serve-indexer` | changes index throughput and where index work runs | event backlog, router CPU, remote-index latency, cache-hit behavior | remote indexer controls are experimental and require both client and serving sides to be configured |
| Multiple router replicas disagree about active sequences | `--router-replica-sync` | can reduce replica-local routing skew | synchronization events, per-router state and decisions, AIPerf | best effort and adds event-plane traffic |
| Follow-up requests should remain on the same worker | `--router-session-affinity-ttl-secs` | can improve continuation locality and reduce repeated prefill | affinity hits, worker assignment, cache reuse, AIPerf | may create skew; bindings synchronize across router replicas only on a best-effort basis |
| Busy workers should be removed from selection sooner | `--active-decode-blocks-threshold`; `--active-prefill-tokens-threshold`; `--active-prefill-tokens-threshold-frac` | may reduce overload tails and failures | busy-worker decisions, eligible worker count, queueing, errors, AIPerf | thresholds use rejection logic and can underuse capacity; absolute and fractional prefill thresholds combine with OR logic |
| Routing should account for an external shared KV cache | `--shared-cache-type`; `--shared-cache-multiplier` | can route toward shared-cache hits without valuing them as device-local hits | shared-cache query and hit evidence, selected workers, AIPerf | experimental and backend-specific; `hicache` is the supported shared-cache type |
| CPU and GPU encoder workers receive the wrong traffic share | `DYN_ENCODER_CUDA_TO_CPU_RATIO` with `device-aware-weighted` routing | changes request allocation in heterogeneous encoder pools | device-class request counts, queueing, latency, AIPerf | inert outside `device-aware-weighted`; set the ratio from measured relative capacity |

## KVBM and KV Transfer

KVBM knobs are active only when the selected engine is configured with the Dynamo KVBM connector. Connector activation
may require multiple engine fields for functionality; keep optional KVBM tuning out of that activation candidate.

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Reusable KV is evicted from GPU and must be recomputed | `DYN_KVBM_CPU_CACHE_GB` or `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` | increases host-tier reuse at the cost of pinned memory and transfer work | configured blocks, GPU-to-CPU offloads, CPU-tier hits, transfer latency, AIPerf | use one sizing form; the CPU tier should not be smaller than the GPU KV tier or write-through churn can regress performance |
| Host KV capacity is insufficient for the reuse window | `DYN_KVBM_DISK_CACHE_GB` or `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS`; `DYN_KVBM_DISK_CACHE_DIR` | adds a larger lower tier and may avoid recompute | mounted device, configured blocks, disk-tier hits, I/O and transfer latency, AIPerf | use one sizing form; disk should not be smaller than CPU, and the path must use a suitable block-backed filesystem |
| Disk filtering discards KV that the workload reuses | `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | may increase disk-tier hits | offload decisions, reuse frequency, disk writes, hit rate, AIPerf | disabling the filter increases writes and SSD wear |
| KV offload or onboard work is serialized | `DYN_KVBM_MAX_CONCURRENT_TRANSFERS` | may expose more transfer parallelism and reduce stalls | concurrent transfers, transfer queue, bandwidth, CPU and memory pressure, AIPerf | too much concurrency can contend with inference or saturate the interconnect |
| Transfers are too fragmented or batches wait too long | `DYN_KVBM_MAX_TRANSFER_BATCH_SIZE` | changes transfer efficiency and wait time | batch sizes, transfer latency and bandwidth, AIPerf | larger batches can increase per-request delay and temporary memory use |
| KVBM uses the wrong transport for the cluster | `DYN_KVBM_NIXL_BACKEND_<BACKEND>` such as UCX, libfabric, or GDS | can engage the intended RDMA or storage data path | KVBM/NIXL startup logs, selected backend, link bandwidth, AIPerf | hardware-, image-, and cluster-specific; do not enable an unavailable backend |
| Multi-GPU MLA workers duplicate lower-tier KV loads | `DYN_KVBM_NCCL_MLA_MODE` | may replace per-rank loads with rank-0 load plus broadcast | NCCL MLA engagement, per-rank load volume, transfer time, AIPerf | only for compatible MLA deployments and requires the expected MPI/NCCL support |

## Local Planner (Conditional)

Local Planner knobs apply only when the current single DGD already contains a Planner and autoscaling behavior is an
explicit optimization objective. Do not add or tune the Planner during a fixed-capacity AIPerf comparison; keep it
disabled, advisory-only, or otherwise hold replicas constant. For an autoscaling experiment, record the full replica
timeline and GPU-seconds.

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Planner cannot change a component's replicas | component `spec.components[*].scalingAdapter` | enables the scaling subresource used by the Local Planner | scaling-adapter object, ownership of replicas, observed scale action | engagement control, not an optimization by itself; do not edit `replicas` directly while the adapter owns it |
| Planner is optimizing the wrong behavior | Planner `mode`; `optimization_target` (`throughput`, `latency`, `load`, `sla`) | changes the signals and policy used for scaling | resolved Planner config, decisions and reasons, replica timeline | mode must match aggregated or disaggregated roles; non-`sla` targets ignore TTFT and ITL targets |
| SLA scaling targets are inappropriate | `ttft_ms`; `itl_ms` | changes the capacity selected to meet latency objectives | Planner estimates, observed TTFT and ITL, replicas, AIPerf | active only with `optimization_target: sla`; tighter targets can consume more GPUs |
| Scaling uses the wrong signal path | `enable_throughput_scaling`; `enable_load_scaling`; `throughput_metrics_source` | changes which observations can trigger a replica decision | source metrics, decision logs, replica timeline | enabled modes interact with `optimization_target`; validate the resolved configuration after model validation |
| Planner allocates outside the intended resource envelope | `max_gpu_budget`; `min_gpu_budget`; `min_endpoint`; `prefill_engine_num_gpu`; `decode_engine_num_gpu` | bounds total capacity and minimum role availability | GPU accounting, per-role replicas, denied decisions | GPU-per-engine values must match the deployed parallel shape; equal min/max budgets permit only tolerance-limited redistribution |
| Planner reacts too slowly or oscillates | `throughput_adjustment_interval_seconds`; `load_adjustment_interval_seconds`; `scheduling.scale_interval_seconds`; `load_scaling_down_sensitivity` | changes reaction time, stability, and scale-down hysteresis | decision cadence, oscillation, SLO recovery time, GPU-seconds | interval divisibility and ordering constraints apply; shorter intervals amplify noisy observations |
| Load-based scaling thresholds do not match capacity | `prefill_scale_up_queue_tokens`; `prefill_scale_down_queue_tokens`; `decode_scale_up_kv_rate`; `decode_scale_down_kv_rate` | changes when prefill or decode scales up and down | queue tokens, KV utilization, decisions, replicas, AIPerf over time | active with load targeting; each up threshold must exceed its corresponding down threshold |
| Forecasts lag or overreact to traffic changes | `load_predictor`; predictor-specific window and noise parameters; `load_predictor_log1p` | changes forecast error, pre-scaling, and oscillation | predicted versus observed load, decision timing, SLO recovery, GPU-seconds | predictor tuning needs a representative time-varying trace, not a short steady-state run |
| Planner decisions need validation before mutation | `advisory` | records decisions without applying scaling | advisory decisions versus expected replica changes | verification mode, not a serving-speed optimization |

## Workload-Specific Controls

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Repeated multimodal inputs redo encoding work | `--multimodal-embedding-cache-capacity-gb`; `--multimodal-embedding-cache-publisher` | can reduce encoder work and multimodal latency | cache publication and hits, encoder load, request latency, AIPerf | publisher is useful with KV-aware routing and is not needed for ordinary aggregated or round-robin paths |
| Repeated remote media fetch and decode work dominates the frontend | `DYN_MULTIMODAL_LOADER_CACHE_GB` | can reduce media load and preprocessing latency | media-cache hits, frontend memory, media processing time, AIPerf | workload-specific; cache capacity consumes frontend memory and is inert without reusable media |

## Evidence to Record

- Exact DGD diff, rendered pod arguments and environment, Dynamo image or version, model, and engine configuration.
- Active replicas, GPUs per replica, node placement, router mode, request and event planes, and Planner state.
- Engagement evidence for the selected mechanism: routing scores or assignments, queue and rejection counts, cache-tier
  hits and transfers, or scaling decisions and replica timeline.
- Valid same-series AIPerf metrics, errors, and resource-use changes. Treat AIPerf as end-to-end outcome evidence, not
  proof of an internal mechanism it cannot observe.

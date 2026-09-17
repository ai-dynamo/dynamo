---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configuration and Tuning
subtitle: Router behavior, event transport, load tracking, and tuning guidance
---

This page explains router behavior and tuning for frontend-embedded and standalone
deployments. For the exact frontend flag names, environment variables, defaults, and
boolean forms, use the [Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router).
For the routing cost model and worker-selection behavior, see
[Routing Concepts](routing-concepts.md).

## Configuration Scope and Precedence

The Frontend configuration is the default for worker sets that do not advertise router settings. A worker set that advertises router configuration replaces that default for requests routed to the set; it does not merge individual settings with the Frontend configuration.

Every replica admitted to a worker set must have the same model deployment card (MDC) checksum after discovery normalization and tokenizer overrides. A worker set is defined by namespace, component, endpoint, model, and worker type. The first valid card observed by a Frontend reserves the set's configuration; replicas with a different checksum receive no traffic and cannot disrupt that configuration.

When a worker set advertises `--router-mode kv`, restate every non-default setting that it needs. An omitted worker flag selects the shared default, not the Frontend's tuned value. This distinction matters most when the Frontend and workers receive different environment variables, such as separate Kubernetes services.

For example, if the Frontend sets `--router-kv-overlap-score-credit 2.5` but a worker set advertises only `--router-mode kv`, the worker set uses the default overlap credit of `1.0`. If both processes inherit the same environment variable, they resolve to the same value. Check the `Activating prefill router` log line to confirm the resolved configuration for each hop.

### Worker-Set Admission and Succession

The first configuration retains its reservation while any matching workers remain,
including during queued construction, failed construction, and retries. Workers
with a different checksum form rejected cohorts. Their registration, removal,
and adapter updates cannot change the incumbent's admissions, serving state,
routing configuration, or retry schedule. A larger rejected cohort has no priority
over the incumbent. The Frontend logs each newly rejected cohort at `ERROR`.

Checksum equality is stricter than equivalent serving behavior. Different advertised
router settings, absent versus explicit defaults, and different model `source_path`
values can produce different checksums even when workers could serve requests the
same way. These differences reject only the newcomer. Supported legacy cards still
join when existing discovery-boundary normalization produces matching checksums;
there is no additional equivalence check or normalized materialization fingerprint.
The MDC checksum algorithm and metadata-cache identity are unchanged.

When the last incumbent worker disappears, the Frontend withdraws its pipeline and
starts a fresh pipeline for the oldest remaining cohort. Duplicate discovery events
and snapshots preserve cohort order. A cohort that disappears completely and later
returns joins the end. Old pipelines cannot route through the successor, even if it
uses the same endpoint or checksum.

For example, a rolling update can serve `model-a` from both `dgd-name-v1` and
`dgd-name-v2`. These versioned namespaces identify separate worker sets, each with
its own admitted configuration and routing pipeline. Their cards do not need to
match each other. Within either set, a replica advertising a different local model
directory is rejected if that difference changes its MDC checksum.

Admission applies to every discovery-managed Frontend routing hop, including
prefill and encoder requests. Each hop uses its committed worker set's selected
card and admitted instances. Prefill routing mode and KV block size come from that
card. Compatible replicas can join without rebuilding the hop; succession replaces
its configuration even if the endpoint is unchanged.

> [!NOTE]
> Selection is local to each Frontend. Frontends that observe conflicting cards in
> different orders may choose different incumbents; no cross-Frontend agreement is
> promised. This intentionally favors serving each Frontend's admitted incumbent
> over serving none. Different local winners are expected; discovery does not
> withdraw service or run a shared election to force agreement.
>
> Frontend readiness reflects committed membership. The KV DC Relay
> evaluates discovery independently and may remain conservative while a Frontend
> serves its incumbent. The shared readiness evaluator produces the same answer
> only for equivalent input units.

## Routing Behavior

- `--router-kv-overlap-score-credit`: Device-local prefix-overlap credit multiplier in the prefill cost calculation. It must be finite and nonnegative. Values greater than `1.0` give overlap extra credit, but the adjusted prefill contribution is clamped at zero. When set to `0`, the router ignores prefix caches and skips creating a local indexer. Defaults to `1.0`.
- `--router-kv-overlap-score-credit-decay`: Decays device-local overlap credit for workers whose active prefill load exceeds the least-loaded eligible worker. `0` disables decay. Defaults to 0.
- `--router-prefill-load-scale`: Scale applied to adjusted prompt-side prefill load after device, lower-tier, and shared-cache credits are subtracted. Defaults to 1.
- `--router-decode-active-request-weight`: Experimental finite, nonnegative block-equivalent decode cost added for each active request on a candidate worker. Defaults to 0.
- `--router-host-cache-hit-weight`: Credit multiplier for host-pinned (CPU offload) prefix overlap, from 0.0 to 1.0. Symmetric to `--router-kv-overlap-score-credit` but applied to the host-pinned tier when a backend exposes CPU offload via a KV connector. Defaults to 0.75.
- `--router-disk-cache-hit-weight`: Credit multiplier for disk/lower-tier (e.g. NVMe-backed) prefix overlap, from 0.0 to 1.0. Defaults to 0.25.
- `--load-aware`: Preset for load-aware KV routing without cache-reuse signals. On the frontend, it implies `--router-mode kv`. It sets `overlap_score_credit=0`, disables KV events and KV reuse assumptions, enables active-block and prefill-token load tracking, disables remote/shared cache indexers, and preserves `--router-prefill-load-scale`, `--router-host-cache-hit-weight`, and `--router-disk-cache-hit-weight`.
- `--router-temperature`: Controls worker selection randomness through softmax sampling of normalized router cost logits. A value of 0 (default) ensures deterministic selection of the lowest-cost worker, while higher values introduce more randomness.
- `--router-conditional-disagg`: **Experimental.** Enables conditional disaggregation in frontend-embedded disaggregated serving. Requires `--router-mode kv`, `--router-kv-events`, separate prefill/decode worker pools, and decode-worker KV event publishing. Use `--router-conditional-disagg-config` for policy settings. See [Conditional Disaggregation](../../../advanced-customizations/conditional-disaggregation.md) for backend requirements and policy tuning.
- `--router-track-prefill-tokens`: Enables prompt-side load accounting in the worker cost model. This should stay enabled if you want queue thresholds, `active_prefill_tokens`, and AIC prefill load decay to reflect prompt work.
- `--router-prefill-load-model`: Selects the router's prompt-side load model. `none` keeps the existing static prompt load accounting. `aic` predicts one expected prefill duration per admitted request and lazily decays only the oldest active prefill request on each worker.
- `--router-queue-threshold`: Optional queue threshold fraction for prefill token capacity. Queueing is disabled by default; setting a numeric value enables it. The router holds incoming requests in a priority queue while all eligible workers exceed `threshold * max_num_batched_tokens`, releasing them when capacity frees up. This defers dispatch rather than rejecting work, so routing decisions use the freshest load metrics at the moment a request is sent to a worker. `nvext.agent_hints.strict_priority` selects an absolute pending-queue tier, while `nvext.agent_hints.priority` adjusts ordering within the configured policy. Must be greater than or equal to 0; use `0.0` for maximum queueing sensitivity. See the SGLang note under [Tuning Guidelines](#tuning-guidelines) for caveats around how `max_num_batched_tokens` is populated on that backend, and see [Priority Scheduling](../../../../use-cases/agents/priority-scheduling.md) for how router priority differs from backend engine priority.
- `--router-queue-policy`: Scheduling policy for the router queue: `fcfs` (default) or `wspt`.
- `--router-policy-config`: Startup-only YAML path for policy-class queues and worker-selection instances. When omitted, `--router-queue-threshold` and `--router-queue-policy` define one synthetic policy class. The equivalent environment variable is `DYN_ROUTER_POLICY_CONFIG`. See [Worker-Selection Policies](#worker-selection-policies) to select a built-in policy, and [Write Custom Routing Strategies](custom-worker-selection.mdx) for the linked-policy schema.

For how queue backpressure differs from candidate filtering and busy-threshold overload handling, see [Router Filtering](worker-filtering.md).

`fcfs` orders by adjusted arrival time (`priority_jump - arrival_offset`) and optimizes tail TTFT.
`wspt` orders by `(1 + priority_jump) / scheduling_cost_tokens` and optimizes average TTFT, where
`scheduling_cost_tokens = max(1, raw_isl_tokens - cached_tokens)`. Both the CLI and policy-class
YAML accept only `fcfs` and `wspt`.

For each policy, the complete pending-queue key is
`(strict_priority, policy_key)`. Higher strict tiers always win; the selected
policy orders requests within a tier.

### Worker-Selection Policies

A worker-selection policy replaces the worker-ranking step of the routing pipeline: it decides which
eligible worker receives a request. Dynamo still owns discovery, eligibility, queueing, reservations,
accounting, and metrics. The built-in selector and its cost model above remain the default.

The Dynamo frontend ships a set of built-in worker-selection policies, so selecting one needs
`--router-policy-config` only — no rebuild, no custom image. They come from the policy catalog the
Python bindings link by default; a build that disables default features, and the standalone EPP,
link no catalog and reject a configured policy type at startup.

| Policy type | Behavior |
|---|---|
| `default` | Dynamo's built-in selector and cost model. Reserved; always available. |
| `dynamo-two-tier-cost-fn` | Ranks on two tiers instead of one additive cost: active-request load first, then device-KV prefix overlap. Prefers the worker holding the largest prefix overlap unless load is badly imbalanced. Thresholds and selection order ported from the experimental SGLang router's `cache_aware_zmq` policy. Thresholds are tunable; the defaults reproduce it exactly. |

Write the instance into the same YAML file that `--router-policy-config` already points at:

```yaml
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  prefill: dynamo-two-tier-cost-fn
  decode: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
```

`aggregated`, `prefill`, `decode`, and `encode` each select a named instance, so prefill and decode
pools can run different policies. An omitted stage falls back to the built-in selector. `name` is
yours to choose; `type` must be one of the policy types above.

```bash
python3 -m dynamo.frontend --router-mode kv --router-policy-config worker-selection.yaml
```

#### Tune a Policy

A policy instance may carry a `parameters` mapping that the policy itself validates at startup.
Omitting it keeps every default, so `dynamo-two-tier-cost-fn` with no `parameters` reproduces the
experimental router exactly. Add any subset to tune it:

```yaml
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_threshold: 0.5
        balance_abs_threshold: 32
        balance_rel_threshold: 1.1
```

| Parameter | Default | Meaning |
|---|---|---|
| `cache_threshold` | `0.5` | Fraction of the request's blocks that must be device-resident on the best worker before the cache tier applies. Compared strictly. Must be in `[0.0, 1.0]`. |
| `balance_abs_threshold` | `32` | Minimum active-request spread before the load tier applies. |
| `balance_rel_threshold` | `1.1` | Minimum ratio of largest to smallest active-request count before the load tier applies. Must be at least `1.0`. |

Both load gates must hold before the load tier displaces the cache tier. Parameters are validated at
startup, so an out-of-range value or an unknown key fails the process immediately, naming the key,
rather than being silently ignored. It selects the least-loaded worker once the active-request spread is greater than 32 and the
largest count is more than 1.1 times the smallest; otherwise it prefers the worker holding the
largest device-KV overlap when that overlap covers more than 50% of the request's blocks.

#### Override the Selection

`DYN_ROUTER_WORKER_SELECTION_POLICY` overrides every stage. `--router-prefill-policy` and
`--router-decode-policy`, and their `DYN_ROUTER_PREFILL_POLICY` and `DYN_ROUTER_DECODE_POLICY`
environment variables, override one stage each. Precedence for a stage is: stage flag, stage
environment variable, `DYN_ROUTER_WORKER_SELECTION_POLICY`, the YAML stage selection, then
Dynamo's built-in selector. There is no YAML-wide default: a stage with no selection falls straight
to the built-in selector, and `worker_selection` rejects unknown keys. Passing `default` explicitly selects the built-in selector
for that scope, which makes it a quick way to A/B a policy against the default.

> [!NOTE]
> If a configured policy type is not linked into the running build, startup fails with the list of
> policy types that are linked. It does not silently fall back to the default selector.

To write your own policy instead of using a built-in one, see
[Write Custom Routing Strategies](custom-worker-selection.mdx).

### Policy-Class Queues

YAML profiles define a matrix from client-requested policy family and
router-observed cache bucket to a physical policy-class queue. Clients send the
requested family through `x-dynamo-meta-policy-class`. The router computes
uncached ISL as `raw ISL - best cached tokens across eligible workers`, selects
the highest matching `uncached_isl_buckets` floor, and resolves the
family/bucket pair to one configured class.

An exact header matching a class with neither `policy_family` nor
`cache_bucket` selects that explicit class directly and intentionally bypasses
cache-derived classification. A recognized family selects that family.
Missing, empty, unknown, or ordinary physical-class names use
`default_policy_family`, so a client cannot bypass cache bucketing by naming a
matrix class directly.

Each class owns a shared FCFS or WSPT heap plus one heap for each exact-worker
lane, along with its busy thresholds, queue limits, quantum, deficit, and
counters. Arbitration compares the shared head with the currently dispatchable
exact-worker lane heads. Absolute and fractional busy thresholds use OR
semantics. A class queues only when at least one threshold is configured and
every eligible worker is busy for that class, but a new arrival cannot bypass
an existing backlog in the same class.

Queue limits are configured per discovered worker endpoint with
`request_queue_limit_per_worker`, `raw_isl_token_queue_limit_per_worker`, and
`cached_token_queue_limit_per_worker`. The effective class-local limit is the
configured value multiplied by the current number of discovered endpoints.
Limits are checked against current usage before adding the incoming request,
so the request that crosses a limit is accepted and the next queued request is
rejected with HTTP 529 and the effective total. Worker removal does not evict
queued requests; new arrivals reject until usage drains or capacity returns.
DRR charges the uncached-token snapshot captured at enqueue, while raw, cached,
and uncached snapshots remain unchanged for limits, WSPT, counters, and later
dispatch. For the ring cursor, deficit charging, weighted bursts, and bounded
bulk-credit behavior, see
[Deficit Round Robin Queue Scheduling](deficit-round-robin.md).

Every matrix class must identify both `policy_family` and `cache_bucket`; a
class with neither field is explicit, while specifying only one is invalid.
Every configured family must have exactly one physical class for every bucket.
Bucket floors begin at zero and increase strictly.
Class, family, and bucket names use metric-safe identifiers.

Profiles resolve in this order: exact model profile, root profile, then the
synthetic single-class fallback. A model profile completely replaces the root
profile; fields, buckets, families, and classes are not inherited. With no
YAML, the router uses a synthetic `default` class and does not compute cache
state for classification. The synthetic class queues only when
`--router-queue-threshold` is set. See the tested
[sample policy](https://github.com/ai-dynamo/dynamo/blob/main/examples/router/policy-class-queues.yaml).

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-policy-config examples/router/policy-class-queues.yaml
```

For a minimal two-class walkthrough showing how one request class can receive
a larger service share without starving another, see
[Prioritize Premium Requests with Policy Classes](deficit-round-robin.md#prioritize-premium-requests-with-policy-classes).

The previous missing-ISL global admission cap is removed. Cache-derived bucket
selection now resolves directly to an ordinary policy class, and that class
owns the queue threshold, ordering, DRR weight, counters, and limits. There is
no separate first-stage admission queue or global cross-class cap.

This is intentionally not behavior preserving. Class limits are worker-scaled
and class-local rather than global; rejection returns the structured
policy-class HTTP 529 response rather than the previous overload 429 path; and
it does not exclude the entire router instance. The previous flat
`default_policy_class` and `uncached_isl_policy_class_tiers` schema is not
accepted, and ordinary physical classes are no longer direct header
overrides. The sample is a Baseten-oriented continuing-session starting point,
not a compatibility profile.

For `--router-mode device-aware-weighted`, set `DYN_ENCODER_CUDA_TO_CPU_RATIO` to the approximate throughput ratio of one non-CPU worker relative to one CPU worker. The default is `8`.

## Session Affinity

Session affinity is disabled by default. On the frontend, set
`--router-session-affinity-ttl-secs` or `DYN_ROUTER_SESSION_AFFINITY_TTL_SECS` to
a value from `1` through `31536000` to enable it, then send
`X-Dynamo-Session-ID` to keep related requests on one worker. Supplying the header
without the TTL option provides session identity but does not enable router affinity.

The first successfully dispatched request binds the session ID to its selected worker and, when available, data-parallel rank. Choose how later requests use that binding with `--router-session-affinity-mode` or `DYN_ROUTER_SESSION_AFFINITY_MODE`:

| Mode | Behavior |
|---|---|
| `hard` | Default. Exact-dispatch to the stored target. If the worker or rank is no longer valid, invalidate the binding and retry normal selection once |
| `soft` | Pass the stored target through the normal selection pipeline as an advisory target. The built-in selector retains it while eligible; a custom policy can choose another worker |

For soft affinity, Dynamo commits a changed binding after dispatch returns a response stream. Selection, setup, or dispatch failure before that point leaves the old binding intact. A later stream error or cancellation does not roll back the rebind. Explicit request targets remain exact in both modes.

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-session-affinity-ttl-secs 300 \
  --router-session-affinity-mode soft
```

Concurrent requests can share a binding. Versioned updates prevent an older concurrent request from replacing a newer soft rebind. Active requests prevent expiry. When a request lease ends after EOF, early drop, error, or cancellation, the idle timer restarts. A missing hard-bound worker or a non-cancellation hard-mode selection, setup, dispatch, or target-validation failure invalidates the binding.

The configured value is the idle timeout. It is independent of
`--router-ttl-secs` and `--router-predicted-ttl-secs`. Omit the session-affinity
option to keep affinity disabled.

With `DYN_LORA_ENABLED`, session affinity is supported in KV mode. It is rejected
at startup with random or round-robin routing; the other router modes are not
LoRA-aware and are rejected independently.

When session affinity is enabled, routers synchronize affinity bindings through the
Runtime event plane. The origin publishes a binding after successful dispatch so
concurrent requests can observe it, then publishes it again when the request lease
ends so peer idle timers restart when the request becomes idle. The extra event
fanout is an intentional tradeoff.

Synchronization is advisory. Each replica owns its local idle TTL and uses the
first live binding it observes. A matching update refreshes that local deadline,
an expired binding can be replaced, and a conflicting live binding is ignored.
Events use the same component scope as active-sequence synchronization, and the
session-affinity subscriber additionally rejects targets outside its local worker
set. Stronger cross-model or worker-role isolation is future work.

Dropped, delayed, or reordered events do not affect request correctness, but can
temporarily reduce affinity. In particular, a long request can outlive a peer's
local TTL, and a dropped lease-completion update can leave peer deadlines out of
sync until a later request republishes the binding.

If the bound worker disappears, Dynamo invalidates the binding so a subsequent
selection can bind an available worker. Router restart clears all bindings. Bindings
received from replicas are not authoritative storage. For strict affinity, configure
the ingress or load balancer to consistently route a session to one frontend, or use
an authoritative external binding store. When hashing at ingress, hash the raw
session header rather than Dynamo's normalized internal `session_id`: canonical
clients send `X-Dynamo-Session-ID`, while agent-native clients use the corresponding
header listed in [Session IDs](../../../../use-cases/agents/session-ids.mdx). Agent-native identity is
normalized only after the request reaches the frontend.

Direct mode still requires the phase-appropriate explicit worker ID on every
affinity request. The stored binding validates that target but does not supply a
missing ID. In disaggregated serving, prefill and decode use separate phase-local
bindings. If no prefill router is active, only the decode or aggregated binding is
created.

Session affinity does not create a backend session or send lifecycle RPCs. There is
no explicit unbind; idle expiry removes only router-local state. The same session
ID is available to tracing and other explicitly configured consumers.

### AIC Prefill Load Model

Use `--router-prefill-load-model aic` when you want prompt-side load tracking to decay the oldest active prefill request using an AIC-predicted duration instead of keeping prompt load static until first token. For the cost-model behavior, see [Prefill Load Modeling](routing-concepts.md#prefill-load-modeling).

Enable it on the frontend like this:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-prefill-load-model aic \
    --aic-backend vllm \
    --aic-system h200_sxm \
    --aic-model-path nvidia/Llama-3.1-8B-Instruct-FP8
```

Required when `--router-prefill-load-model=aic` is enabled:

- `--router-mode kv` on the frontend
- `--router-track-prefill-tokens`
- `--aic-backend`
- `--aic-system`
- `--aic-model-path`

Optional AIC knobs:

- `--aic-backend-version`: pinned AIC database version; if omitted, Dynamo uses a backend-specific default
- `--aic-tp-size`: tensor-parallel size for the modeled backend; defaults to `1`
- `--aic-moe-tp-size`: MoE tensor-parallel size for models that require AIC MoE parallelism
- `--aic-moe-ep-size`: MoE expert-parallel size for models that require AIC MoE parallelism
- `--aic-attention-dp-size`: attention data-parallel size for models that require AIC MoE parallelism

For MoE models, these values must satisfy AIC's parallelism constraint:
`aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size`.
For Kimi-style TP-only MoE runs, use `--aic-moe-tp-size` equal to `--aic-tp-size`,
`--aic-moe-ep-size 1`, and `--aic-attention-dp-size 1`.

## KV Event Transport

- `--no-router-kv-events`: Disables KV event tracking. By default, the router consumes KV events to monitor block creation and deletion from workers that publish them. When disabled, the router predicts cache state from routing decisions. Predicted entries use TTL retention by default; the experimental local LRU policy is described below.

## Topology-Aware KV Transfer

Topology-aware KV transfer is configured on workers through runtime metadata, not with frontend router flags. In Kubernetes, use `spec.experimental.kvTransferPolicy` on the `DynamoGraphDeployment`; the operator injects the worker environment and topology files. Outside Kubernetes, set `DYN_TOPOLOGY_ENABLED`, `DYN_TOPOLOGY_MOUNT_PATH`, `DYN_KV_TRANSFER_DOMAIN`, and `DYN_KV_TRANSFER_ENFORCEMENT` on workers. Set `DYN_KV_TRANSFER_PREFERRED_WEIGHT` only when enforcement is `preferred`.

For the full runtime contract and routing behavior, see [Topology-Aware KV Transfer](topology-aware-kv-transfer.md).
For the Kubernetes configuration fields, see the [KvTransferPolicy API](../../../../reference/kubernetes-api/full-api-reference.mdx#kvtransferpolicy).

## Block Tracking

- `--no-router-track-active-blocks`: Disables tracking of active blocks used for ongoing generation or decode phases. Disable this when routing to workers that only perform prefill.
- `--router-track-output-blocks`: **Experimental.** Enables tracking of output blocks during generation. When enabled, the router adds placeholder blocks as tokens are generated. With an expected output sequence length (`agent_hints.osl` in `nvext`), fractional decay applies to output blocks and the structurally exclusive prompt suffix; shared prompt blocks retain full weight. For the cost-model behavior, see [Decode Load Modeling](routing-concepts.md#decode-load-modeling).
- `--no-router-assume-kv-reuse`: When tracking active blocks, disables the assumption of KV cache reuse. This is useful in disaggregated setups where transferred blocks are not actually deduplicated on the decode side.
- `--no-router-track-prefill-tokens`: Disables prompt-side prefill token accounting in the router's active load model. Use this for decode-only routing paths where prompt processing already happened elsewhere.
- `--router-replica-sync`: Disabled by default. Enables best-effort Runtime event-plane synchronization of KV active-sequence state. Session-affinity synchronization is independent and starts when `--router-session-affinity-ttl-secs` is set.
- `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS`: Environment-only request-liveness duration. The default is `300` seconds for both implementations. Legacy selection-service and standalone slot-tracker state expires by absolute age, approximately five to six minutes after admission regardless of output progress. The embedded `KvRouter` uses the duration as its shared CLOCK scan interval. Output progress grants a lease one second chance, so idle cleanup occurs approximately five to ten minutes after the last progress touch. Replica mirrors refresh only from synchronized lifecycle events. Each router expires local and mirrored copies independently without publishing `Free`; expiry removes only that router's scheduler state and local approximate-LRU references. Explicit lifecycle completion publishes `Free` and remains idempotent after local expiry. This request-liveness policy is separate from approximate-cache retention TTL and does not turn best-effort synchronization into authoritative state.

### Tracking Hash Identities

**Experimental.** Set `--router-tracking-hash keyed-xxh3-v1` to make
router-derived active-sequence identities depend on a provider key. The default
`public-xxh3-v1` mode preserves the existing public XXH3 identities.

Keyed mode requires `--router-tracking-key-file` to name a file containing
exactly 32 raw bytes and `--router-tracking-key-id` to contain a nonempty key
epoch. The corresponding environment variables are
`DYN_ROUTER_TRACKING_HASH`, `DYN_ROUTER_TRACKING_KEY_FILE`, and
`DYN_ROUTER_TRACKING_KEY_ID`. Invalid or unreadable key configuration stops
startup instead of falling back to public hashing.

The router derives independent block and chain XXH3 seeds from one keyed BLAKE3
digest for each request scope. The scope includes the algorithm version, key ID,
model, routing group, block size, normalized `cache_salt`, LoRA adapter, and
Eagle mode. Multimodal identity remains part of the canonical bytes for each
block. Seeded XXH3 is not a pseudorandom function or message authentication
code, so keep tracking hashes and the APIs that accept them on a trusted
internal plane.

Public block hashes continue to drive primary-indexer lookups. Keyed sequence
hashes drive active tracking, reservations, prompt membership, and projected
load. Engine hashes, universal positional lineage hashes, KV events, and
standalone indexer APIs do not change. `--no-router-assume-kv-reuse` continues
to use random tracking identities while retaining public indexer hashes.

Selection requests and standalone slot-tracker calls that supply precomputed
sequence hashes remain trusted inputs. Dynamo does not attach or negotiate an
algorithm or key epoch in HTTP requests, selection payloads, tracker lifecycle
events, or replica messages. Initialize every producer with the same algorithm,
key, and key ID. To rotate the key, change the key and key ID together, restart
all producers, and flush or recreate derived tracker state before resuming
traffic. Mixed epochs are not detected.

## KV Indexer / Approx KV Indexer

- `--router-ttl-secs`: Time-to-live in seconds for blocks in the router's local cache predictions. Defaults to 120.0 seconds when `--no-router-kv-events` is used.
- `--router-approximate-cache-policy`: Retention policy for a local approximate primary indexer. `ttl` is the default. Experimental `lru` models the physical KV capacity advertised by each worker data-parallel rank, retains complete canonical prompt and output blocks, and evicts the least recently used unreferenced copies under pressure. It requires `--no-router-kv-events`. Remote and served approximate indexers fall back to TTL; the predict-on-route side indexer is always TTL-only. The equivalent environment variable is `DYN_ROUTER_APPROXIMATE_CACHE_POLICY`.
- Approximate-LRU mutation lanes currently use unbounded queues. Bounded backpressure is deferred while this policy remains experimental.
- `--router-event-threads`: Number of KV indexer worker threads (default: 4). Values greater than 1 use the concurrent radix tree for event-driven routing, approximate routing with `--no-router-kv-events`, and the predict-on-route side indexer.
- `--router-predicted-ttl-secs`: Enables predict-on-route with this TTL in seconds for entries in a local side indexer. Requires KV events; omit to disable. When enabled, the router feeds each routing decision into the side indexer and scores each worker with the larger overlap from the primary indexer and the local side indexer. Independent of `--router-ttl-secs`; kept short so decisions the engine never confirms (cancelled requests, prefill failures) age out quickly.

### When to use `--router-predicted-ttl-secs`

Without this setting, an event-driven router depends entirely on engine KV events to learn which worker now holds which prefix. That works for steady-state traffic, but creates a race when many sibling requests arrive in a single batch — for example, 16 problems × 4 samples each with a shared system prompt, or any parallel-sampling / best-of-N workload. No engine has emitted a "block stored" event yet, so the router scores every sibling with zero overlap and round-robins them across workers. The prefix then gets prefilled on every worker instead of being reused.

Setting `--router-predicted-ttl-secs 5` makes the router record each routing decision into a secondary, short-TTL approximate indexer. When the next sibling is scored, the router queries both indexers and takes the per-worker max overlap, so siblings see the first sibling's prefix immediately and pin to the same worker. The primary event-driven indexer is untouched — engines compute their sequence hashes with salts and cryptographic digests the router cannot reproduce, so inserting router-computed hashes into the primary would key the same physical block under two different hashes and pollute the tree. Running the two trees in parallel sidesteps that entirely; the side tree has a short TTL and its entries simply expire once the primary takes over.

Do not combine this setting with `--no-router-kv-events`, including when the approximate primary is remote: approximate mode already inserts on routing decisions by construction, and running a second approximate side indexer is redundant. With `--use-remote-indexer` and KV events enabled, the side indexer remains local to the consumer router while the remote indexer remains the shared primary view. If a router also serves an indexer for other routers, the side indexer is still local only; it is never served or consumed as the remote primary.

To implement KV event publishing for custom inference engines, see [KV Event Publishing for Custom Engines](../../../advanced-customizations/writing-custom-backends/publish-kv-events.md).
For details on per-request agent hints (`priority`, `osl`, `speculative_prefill`), see [NVIDIA Request Extensions (`nvext`)](../../../additional-resources/nvidia-request-extensions-nvext.md#agent-hints).

## Tuning Guidelines

`--router-kv-overlap-score-credit` is the primary knob for cache reuse. It credits device-local prefix overlap against the prefill load and must be finite and nonnegative. Higher values steer requests toward workers with better cache overlap and reduce TTFT. Values above `1.0` can saturate the adjusted prefill contribution at zero, so use them deliberately: additional credit cannot make that contribution negative. Lower values distribute load more evenly and reduce ITL. The default of `1.0` is a reasonable starting point. For direct router APIs and EPP integrations, the same router policy can be overridden per request with `router_config_override.overlap_score_credit`; it is not an `nvext.agent_hints` field.

Use `--router-kv-overlap-score-credit-decay` to reduce that device-local credit when a worker has more active prefill work than the least-loaded eligible worker. This helps prevent busy, cache-rich workers from repeatedly winning while newly autoscaled or lightly loaded workers receive too little traffic. The router normalizes the excess active prefill blocks by the incoming request size and multiplies the configured overlap credit by `1 / (1 + decay * normalized_excess)`. For example, a decay of `1` halves device credit at one request-equivalent of excess prefill load. Host, disk, and shared-cache credits are unchanged. This setting requires prefill-token tracking to have an effect and defaults to `0`.

Use `--load-aware` when you want the KV scheduler's active load model without prefix/cache reuse. This is equivalent to using KV mode with overlap credit set to 0, KV events disabled, KV reuse assumptions disabled, active load tracking enabled, and shared-cache routing disabled. `--router-prefill-load-scale` remains available to tune prompt-side load relative to decode blocks.

Deprecated: `--router-kv-overlap-score-weight`, `--kv-overlap-score-weight`, `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT`, and `DYN_OVERLAP_SCORE_WEIGHT` are still accepted, but emit deprecation warnings. Nonzero legacy values map to `prefill_load_scale` to preserve existing behavior without changing overlap credit. A legacy value of 0 maps to both `prefill_load_scale=0` and `overlap_score_credit=0`, which preserves the old no-overlap/no-indexer behavior. If a deprecated overlap score weight is still present, it takes precedence over the newer prefill load scale field; a legacy value of 0 also takes precedence over the newer overlap credit field. When migrating to `--router-prefill-load-scale` or `DYN_ROUTER_PREFILL_LOAD_SCALE`, remove the deprecated flag, env var, or JSON field from the deployment config. Use `--router-kv-overlap-score-credit` or `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` only when you mean to tune the cache-overlap credit itself.

When migrating the deprecated overlap score weight, use `--router-prefill-load-scale` to preserve its scaling role. Tune `--router-kv-overlap-score-credit` separately only when you intend to change device-local cache credit; values above `1.0` are supported, with adjusted prefill cost clamped at zero.

Use `--router-prefill-load-scale` when prompt-side load should count more or less than decode-side block load after cache-hit credits are applied. The final score is `prefill_load_scale * adjusted_prefill_blocks + potential_decode_blocks + decode_active_request_weight * active_requests`.

Use `--router-decode-active-request-weight` when decode forward-pass time depends more on the number of active requests than on their resident KV footprint. The value is measured in block-equivalent cost per active request. For example, a weight of `32` makes four active requests contribute the same routing cost as 128 potential decode blocks. The router captures active-request count with the same worker-load snapshot used for prefill tokens and potential decode blocks, so enabling the term does not add a second slot-tracker lookup.

This setting is experimental and defaults to `0`, which preserves block-only decode scoring. A positive value trades some KV locality for batch-size balance and can help model, runtime, and hardware combinations near a compute-bound roofline knee, including some MLA or MTP configurations. It can regress throughput, TTFT, and ITL when decode remains primarily memory-bound, so benchmark representative traffic and start with a small weight before increasing it.

Use `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight` when the backend exposes lower-tier prefix cache via a KV connector (for example, vLLM's `OffloadingConnector` for CPU offload, or a disk-backed tier). These multipliers control how much each lower-tier hit credits against the prefill load, mirroring the role of `--router-kv-overlap-score-credit` for the device tier. A worker holding a full prefix in CPU offload gets `host_cache_hit_weight * matched_blocks` credit against its prefill cost; raising the weight makes the router more willing to route prefix-matched requests to that worker even if a different worker has a partial device-local match.

Use `--no-router-kv-events` when you are not confident that your backend engine emits KV events correctly. In this mode the router falls back to approximate routing. Keep the default TTL policy unless you are explicitly testing the experimental per-rank capacity-bounded LRU with workers that publish a positive `total_kv_blocks` value.

Use `--router-predicted-ttl-secs 5` when the workload fires bursts of sibling requests with shared prefixes — parallel sampling, best-of-N, agent fan-out. It closes the window between the routing decision and the engine's first "block stored" event so siblings co-locate on the worker the first sibling picked. See the configuration section above for the side-indexer mechanics.

Use `--no-router-assume-kv-reuse` in disaggregated setups where the decode worker does not reuse transferred KV cache blocks. Without this flag, the router undercounts decode blocks when duplicates exist, leading to inaccurate load estimates.

Use `--no-router-track-prefill-tokens` when a router is serving decode-only traffic and prompt processing has already completed elsewhere. This keeps decode routing decisions focused on decode-side load instead of briefly charging prompt tokens to the decode worker after handoff.

Use `--router-track-output-blocks` when your workload is output-heavy and you want the router to account for output-side KV cache growth in load balancing. If you also pass `nvext.agent_hints.osl` per request, the router applies fractional decay to output blocks and the structurally exclusive prompt suffix so that requests nearing completion contribute less future load. Shared prompt blocks retain full weight. See [Decode Load Modeling](routing-concepts.md#decode-load-modeling) for the cost-model details.

`--router-queue-threshold` controls when incoming requests are held in a priority queue. The router waits while all eligible workers exceed `threshold * max_num_batched_tokens`, then releases work as capacity frees up. A lower value queues earlier; `0.0` queues as soon as all eligible workers have any active prefill tokens. Priority hints have no router-level effect when requests do not enter this queue.

This threshold delays dispatch. It does not remove workers from the candidate set; for that distinction, see [Router Filtering](worker-filtering.md).

Use `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS` when queued requests may wait long enough for worker cache state to materially change before dispatch. The default is `10` seconds; set it to `0` to disable dequeue-time overlap refresh.

**Note for the SGLang backend.** In Dynamo v1.1.0 and later, the value the SGLang worker publishes for `max_num_batched_tokens` in its Model Deployment Card depends on the server args:

- If `--max-prefill-tokens` is set, MDC's `max_num_batched_tokens` equals that value (the per-step prefill window — the value most users expect).
- If `--max-prefill-tokens` is not set, MDC's `max_num_batched_tokens` falls back to `max_total_num_tokens` from SGLang's `scheduler_info`, which is the **total KV cache pool** in tokens. On large GPUs with high `mem-fraction-static` the pool can be hundreds of thousands of tokens — much larger than `chunked-prefill-size`.

The threshold is applied as `active_tokens > threshold * max_num_batched_tokens`, so this fallback inflates the effective denominator and a threshold like `1.0` may effectively never queue. To get the originally intended "fraction of the per-step prefill window" semantics on SGLang, either set `--max-prefill-tokens` explicitly on the SGLang backend so the MDC value matches the prefill window, or use a much smaller `--router-queue-threshold` (for example `0.1`) to compensate for the inflated denominator.

Use `--router-prefill-load-model aic` when you want prompt-side load tracking to decay the oldest active prefill request using an AIC-predicted duration instead of keeping prompt load static until first token. This requires `--router-track-prefill-tokens` and the shared `--aic-*` config; see [AIC Prefill Load Model](#aic-prefill-load-model) for the full flag set and [Prefill Load Modeling](routing-concepts.md#prefill-load-modeling) for the cost-model details.

Use `--router-queue-policy wspt` when your workload has a mix of short and long requests and you want to minimize average TTFT. Use the default `fcfs` when you want to minimize tail TTFT.

## Prometheus Metrics

The router exposes Prometheus metrics on the frontend's HTTP port (default 8000) at `/metrics`:

- **Router request metrics** (`dynamo_component_router_*`): Registered via the component's metrics hierarchy and exposed on the frontend via the `drt_metrics` bridge. In KV mode they are populated per request; in non-KV modes they are registered with zero values. The standalone router also registers these metrics, available on `DYN_SYSTEM_PORT` when set.
- **Routing overhead metrics** (`dynamo_router_overhead_*`) and **per-worker gauges** (`dynamo_frontend_worker_*`): Registered on the frontend's own Prometheus registry. These are frontend-only and not available on the standalone router.

For the full list of router metrics, see the
[Metrics Catalog](../../../../reference/observability/metrics-catalog.mdx#router-metrics).

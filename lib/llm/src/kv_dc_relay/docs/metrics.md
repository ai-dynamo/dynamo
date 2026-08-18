<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay: Metrics

Prometheus metrics exported by the Relay WAN transport
(`transport/metrics.rs`), registered on the owning component's registry and
sampled through its update callback. One metrics owner exists per component;
concurrent relay instances share it through a lease bound to the active
`RelayIdentity`.

## Stream health

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `kv_dc_relay_subscribers` | gauge | `stream` = `catalog` \| `pool` \| `readiness` \| `load` | Current subscriber count per stream type. |
| `kv_dc_relay_subscriber_lagged_total` | counter | `stream` | Streams terminated after bounded-queue lag. The consumer must resubscribe; sustained growth means an undersized queue or a slow consumer. |
| `kv_dc_relay_subscriber_limit_rejected_total` | counter | `stream`, `scope` = `total` \| `per_pool` \| `initialized_hub` | Subscriptions rejected at a configured resource limit (`RESOURCE_EXHAUSTED`). `per_pool` and `initialized_hub` scopes exist only for `stream="pool"`. |
| `kv_dc_relay_pool_heartbeats_total` | counter | — | Application-level heartbeats emitted across pool streams. A flat line while pool subscribers exist indicates a stalled publication path. |
| `kv_dc_relay_load_updates_total` | counter | — | Complete pool-load windows emitted. Expected rate: one per configured load window. |

## Producer state

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `kv_dc_relay_catalog_pool_count` | gauge | — | Pools currently in the producer catalog (materialized endpoints only). |
| `kv_dc_relay_requested_publication_hub_count` | gauge | — | Pools whose lazy publication hub has been requested by at least one subscriber. |
| `kv_dc_relay_initialized_publication_hub_count` | gauge | — | Per-pool hubs holding an initialized CKF mirror (each consumes memory; bounded by `max_initialized_pool_hubs`). |
| `kv_dc_relay_ready_publication_hub_count` | gauge | — | Hubs with a contiguous, ready CKF mirror. |
| `kv_dc_relay_idle_publication_hub_count` | gauge | — | Ready hubs with no active subscribers (candidates for reclamation pressure). |
| `kv_dc_relay_terminal_publication_failures_total` | counter | — | Pool generations fenced after a terminal publication failure. Any growth is abnormal: each increment is a pool restart (catalog withdrawal + new generation). |

## Pull diagnostics

Builds with `ckf-diagnostics` include exact per-actor-generation counters in
the Relay diagnostic response. These fields are not Prometheus time series.

| Field | Type | Meaning |
| --- | --- | --- |
| `aggregation.parent_not_found` | counter | Store events rejected because their engine-specific parent ID was absent from that source's lineage. The source is rejected instead of interpreting the block as a new root. |
| `memory.source_lineage_capacity` | entries | Allocated entry capacity across the per-source external-to-canonical lineage tables. |
| `memory.canonical_owner_capacity` | entries | Allocated entry capacity of the canonical ownership table, including owners whose block was omitted from the physical filter. |
| `memory.resident_capacity` | entries | Allocated entry capacity of the exact set of canonical hashes currently resident in the physical filter. |

## Serving topology

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `kv_dc_relay_serving_topology_count` | gauge | `state` = `ready` \| `unavailable` \| `unknown` | Current `(namespace, model)` topology entries by readiness state. `unknown` stuck non-zero after startup means an endpoint whose instance watch never became authoritative. |

## Load quality

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `kv_dc_relay_degraded_load_coverage_pool_count` | gauge | — | Pools where at least one load signal lacks complete rank coverage (`observed < expected`). Consumers see the same honesty markers per entry; non-zero here means load-based decisions run on partial data. |

## Security

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `kv_dc_relay_tls_expiry_timestamp_seconds` | gauge | `material` = `server_cert` \| `client_ca` | Earliest certificate expiry per TLS material. Alert well before `now() > value`. |

## Suggested alerts

- `increase(kv_dc_relay_terminal_publication_failures_total[10m]) > 0` — pool
  generations are being fenced; inspect relay logs for the failing pool.
- `kv_dc_relay_serving_topology_count{state="unknown"} > 0` for more than a few
  minutes after startup — availability watch never became authoritative.
- `rate(kv_dc_relay_subscriber_lagged_total[5m]) > 0` — a consumer cannot keep
  up; check WAN throughput and queue sizing.
- `kv_dc_relay_tls_expiry_timestamp_seconds - time() < 30 * 86400` — certificate
  expires within 30 days.

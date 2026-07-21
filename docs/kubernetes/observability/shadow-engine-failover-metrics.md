---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Shadow Engine Failover Metrics
subtitle: Prometheus metrics and the Grafana dashboard for observing GMS shadow-engine failover.
---

> ⚠️ **Experimental**: These metrics accompany the experimental
> [Shadow Engine Failover](../shadow-engine-failover.md) feature; names and the
> dashboard may change as the feature settles.

## Overview

Each shadow-mode engine emits a small set of self-reported Prometheus metrics
describing its position in the failover lifecycle (`init → standby → waking →
active`), plus counters for real failovers. They are exposed on the engine's
system `/metrics` surface (no extra port) and scraped by the standard
`dynamo-worker` PodMonitor. A Grafana dashboard renders them per
DynamoGraphDeployment (DGD).

## The dashboard

The **Dynamo · GMS Shadow-Failover** dashboard ships as a ConfigMap in the
platform Helm chart (labeled `grafana_dashboard`), so any kube-prometheus-stack
Grafana with the dashboards sidecar auto-imports it — no manual step. It is
**multi-DGD aware**: two template variables, **Namespace** and **DGD**, select a
deployment, and every panel filters to that deployment's failover engines. The
DGD name comes from the `nvidia.com/dynamo-graph-deployment-name` pod label,
exposed as the `dynamo_graph_deployment` metric label by the PodMonitor.

## Metric catalog

All metrics are prefixed `dynamo_component_engine_failover_` and labeled
`engine_id`, `model`, `dynamo_component` (plus `namespace`, `pod`,
`dynamo_graph_deployment` added by the PodMonitor).

| Metric | Type | What it tells you |
|---|---|---|
| `_state` | gauge (1-hot over `state`) | Current lifecycle state. **Source of truth** — `active`/`standby` counts derive from it. A dead engine has no state (series goes stale); "dead" is inferred, not self-reported. |
| `_state_entered_timestamp_seconds` | gauge | When the current state was entered; `time() - value` is the state age (a climbing age on a stuck engine is a wedge signal). |
| `_last_state_duration_seconds{state}` | gauge | Duration of the most recent occupancy of each state; `{state="waking"}` is the wake/switch time. |
| `_transitions_total{from_state,to_state}` | counter | Every transition, including ones too brief for the sampled `_state` gauge to catch (e.g. a sub-second `waking`). |
| `_switch_attempts_total` | counter | Real failover promotions attempted (a shadow that won a *contended* lock). The initial bootup acquires the lock immediately and is **not** counted. |
| `_switch_success_total` | counter | Failover promotions that completed and began serving. Derived failures = `attempts − success`. |

### Derived views

- **Active engines** = `count(_state{state="active"} == 1)` — outage (`0`) / split-brain (`>1`).
- **Shadow ready** = `count(_state{state="standby"} == 1)` — `0` is degraded (one failure from outage).
- **Failovers / failures** — use range queries (`increase(..[$__range])`) so a failover
  by an engine that has since died still counts; an instant `sum` would drop it when the series goes stale.

## Durability: counters that can outlive a scrape

Prometheus is pull-based, so a counter increment lives in process memory until
the next scrape. A failed wake often *is* what kills the process, within the
scrape interval — so the increment can die unscraped. This is not unique to
failover; it is the general "event lost before scrape" problem for ephemeral
work.

The switch counters hedge against it with **write-through persistence**: each
increment is also written (atomic temp+rename+fsync) to a per-engine file in the
shared GMS directory, and reloaded on process start. So the count survives the
death and is re-exposed when the engine (or its restart) comes back — the dying
process never has to report its own failure.

Two limits remain, and both are covered by other signals rather than by the
counter:

- An increment that dies **before its very first scrape** was never sampled, so
  even a range query can't recover it until the engine re-exposes it on restart.
- An **uncaught kill** (OOM/SIGKILL) may not reach the increment at all.

For both, the **k8s panels** (container restarts, `OOMKilled`, `CrashLoopBackOff`
from kube-state-metrics) are the immediate witness that a cutover failed — the
engine can't self-report its own death. The canonical fuller answer for
attributed, scrape-independent capture is a push mechanism (an event/span emitted
on the failure path); that is planned as a follow-up.

## Diagnostics

Beyond the failover metrics, the dashboard overlays cluster signals for the
selected DGD's pods: **container restarts / ready / phase** (kube-state-metrics)
and **GPU framebuffer used** (DCGM). Note the shadow shares one weight copy via
GMS, so per-engine GPU memory should not be summed.

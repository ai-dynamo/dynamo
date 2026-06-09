<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DIS-2172 — Real end-to-end event-plane THROUGHPUT + LOSS bench

This harness measures how well the THREE REAL event-plane consumers keep up
under a request-driven load. Unlike the latency bench (synthetic
`event_plane_bench_sub`), it instruments the actual consume loops:

| # | event | producer | real consumer (instrumented) | counter site label |
|---|-------|----------|------------------------------|--------------------|
| 1 | `kv-events` | mocker workers | KV Router indexer in `dynamo.frontend --router-mode kv` | `kv-events` |
| 2 | `forward-pass-metrics` (FPM) | mocker workers | `dynamo.common.recv_forward_pass_metrics --mode throughput` | `fpm` |
| 3 | `active_sequences` | router replica | peer router replicas (`--router-replica-sync`, needs s≥2) | `active_sequences` |

Each consumer carries a receive-side counter (`RecvCounter` in Rust /
`get_throughput_stats()` in Python), gated by `DYN_BENCH_COUNT=1`, which emits a
per-window JSON line to stderr:

```
{"dis2172_recv":"window","site":"kv-events","window_idx":2,"received":1200,"gaps":0,
 "events_per_sec":120.0,"drop_rate":0.0,"n_publishers":16,"total_received":..., ...}
{"dis2172_recv":"final","site":"kv-events","received":...,"gaps":...,"drop_rate":...}
```

`bench.py` scrapes those lines per component (a router process emits BOTH
`kv-events` and `active_sequences` lines; the scraper filters by the `site`
field). It is **clock-free**: loss is transport-level `EventEnvelope.sequence`
gaps; there are no timestamps/latency.

## Run

```bash
# Local smoke (single host, loopback):
DYN_BENCH_PY=/opt/dynamo/venv/bin/python \
python bench.py --launcher local --transports nats --workers 4 --subs 2 \
  --duration 30 --out results/smoke

# Multi-node (slurm + pyxis):
export DYN_BENCH_IMAGE=<registry>:<tag>     # built by build_image.sh
sbatch bench.sbatch                          # or run bench.py --launcher slurm directly
```

`--subs` is the **router-replica count** (s). s=1 measures kv-events + FPM; s≥2
additionally activates `--router-replica-sync` so `active_sequences` flows
between replicas (loadgen rotates requests across all s frontend ports).

Resource cost (TCP-connection count, per-class CPU/RSS/fd) is sampled per cell
by `ressample.py` (host-level, sees enroot processes), folded into each row.

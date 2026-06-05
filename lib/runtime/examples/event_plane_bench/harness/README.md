<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DIS-2172 event-plane benchmark harness

Unified single-node **and** multi-node driver for the Dynamo event-plane
**NATS vs ZMQ** comparison. It pairs with the `event_plane_bench_sub` Rust
subscriber in the parent directory (`../src/bin/event_plane_bench_sub.rs`): the
harness produces real event-plane traffic and the Rust subscriber does the
actual measurement.

## What it measures

For each sweep cell `(transport × workers p × subscribers s × topic)` the
harness starts a Dynamo frontend + `p` mocker workers (event-plane publishers),
drives request load so the workers emit real KV-cache / forward-pass events, and
runs `s` instrumented `event_plane_bench_sub` subscribers per topic. Each
subscriber writes a JSON result that the harness aggregates into `summary.json`;
`report.py` turns that into tables + plots.

Three metrics come out of every run:

- **Throughput (events/s)** and **drop_rate (seq-gap loss)** — the **primary,
  no-PTP metrics**. They are subscriber-local (no cross-node clock needed), so
  they are valid on multi-node runs even without PTP-synced clocks. Run the
  saturation sweep with `--speedup-ratio 0` (unthrottled mockers) to push the
  transports to their limit and surface back-pressure / fan-out cost.
- **One-way publish→deliver latency (ns, p50/p90/p95/p99)** — only trustworthy
  **single-host** (`--launcher local`, `CLOCK_MONOTONIC`). Cross-node latency
  requires PTP/NTP-synced clocks; without that, treat latency as N/A and rely on
  throughput + drop_rate. `report.py` annotates the tables accordingly.

> Single-host runs go over `lo` (no NIC): they measure the transport *software*
> path, not the hardware cost of brokerless O(p×s) ZMQ fan-out. A multi-node run
> (real NIC) is still required before a default-transport decision.

## Files

These files form the unified flow and must stay **together in this directory**
— `bench.py` derives sibling paths (`loadgen.py`, `zmq_broker.py`, `report.py`)
from its own location:

| file             | role                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `bench.py`       | orchestrator: `LocalLauncher` / `SlurmLauncher`, pyxis container mode, sweeps `p × s × transport × topic`, `--speedup-ratio` |
| `report.py`      | latency + throughput + drop_rate tables + plots, with clock-aware caveats |
| `loadgen.py`     | concurrent request load generator (keeps workers busy → events)      |
| `zmq_broker.py`  | XSUB/XPUB proxy for the optional `zmq-broker` transport              |
| `bench.sbatch`   | pyxis multi-node SLURM wrapper (host orchestrator + containerized tasks) |
| `build_image.sh` | verified recipe to build + push the instrumented Dynamo image       |

## Single-node usage

Requires a running `etcd` (2379) + `nats` (4222), the Dynamo Python env
(`dynamo.frontend` / `dynamo.mocker`), and the built subscriber binary.

```bash
# build the subscriber once (from the example crate, parent dir)
cargo build --release \
  --manifest-path ../Cargo.toml --bin event_plane_bench_sub

# point the harness at the binary + python (defaults are overridable via env)
export DYN_BENCH_SUB_BIN="$PWD/../target/release/event_plane_bench_sub"
export DYN_BENCH_PY="$(which python)"

python bench.py --launcher local \
  --transports nats,zmq \
  --workers 1,4,16,64 \
  --subs 1 \
  --topics kv-events,forward-pass-metrics \
  --speedup-ratio 10

python report.py --in results/run     # -> results/run/report.md + plots
```

Useful knobs: `--speedup-ratio 0` (unthrottled saturation test), `--subs 1,2,4`
(subscriber/router-replica sweep), `--trials N`, `--duration`, `--concurrency`.

## Multi-node usage (SLURM + pyxis)

Multi-node spreads the `p` mocker workers across the allocated nodes so their
events cross the real NIC. Build + push the instrumented image **once**, then
submit the batch job:

```bash
# 1) on a build host: build + push the instrumented image
PUSH_TO=<registry>/dynamo:dis2172-dev ./build_image.sh

# 2) submit (sbatch inherits the env)
export DYN_BENCH_IMAGE=<registry>/dynamo:dis2172-dev
sbatch bench.sbatch
```

`bench.sbatch` runs `bench.py --launcher slurm` on node0 with the host python,
`srun`s every task (etcd/nats/frontend/mocker/loadgen/subscriber) into the
container via pyxis, then runs `report.py`. Override the sweep with the env vars
documented at the top of `bench.sbatch` (`WORKERS`, `SUBS`, `TRANSPORTS`,
`TOPICS`, `GPUS_PER_NODE`, `SPEEDUP_RATIO`, `TRIALS`). For an NVL72-style tray,
use `--gpus-per-node` = workers per node (one mocker per GPU).

> Multi-node latency depends on PTP/NTP clock sync across nodes. `bench.sbatch`
> records the per-node clock offset to `clocksync.txt`; the throughput +
> drop_rate metrics remain valid regardless.

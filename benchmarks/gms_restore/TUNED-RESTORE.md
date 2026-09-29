<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Full restore: resident agent, buffer tuning and CPU

Nine successful restores on the DRA B200 node, rotating original buffers, tuned buffers and tuned buffers with more GMS CPU. Snapshot agent and PageBroker were Ready before every pod-create timer; neither daemon starts inside this measurement. PageBroker integration was unchanged. All cases restored the default-communication GLM capture and passed Berlin generation before Ready and Rayleigh generation afterward.

| Per-rank configuration | n | Pod → Ready mean (range), s | GMS span, s | Rank initialization, s | Rank load + commit, s | CRIU, s | CUDA, s | Restore operation, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 MiB, CPU 1/8 | 3 | 27.519 (26.575–28.580) | 21.851 | 0.318 | 20.040 | 7.148 | 8.339 | 18.252 |
| 128 MiB, CPU 1/8 | 3 | 27.053 (26.656–27.506) | 20.188 | 0.320 | 18.391 | 6.688 | 9.926 | 19.445 |
| 128 MiB, CPU 8/16 | 3 | 27.479 (27.395–27.587) | 19.841 | 0.281 | 18.096 | 6.526 | 9.748 | 18.979 |

The 128 MiB buffers reduced mean GMS span by 1.663 s, but mean end-to-end readiness improved only 0.466 s. Increasing GMS CPU request/limit from 1/8 to 8/16 changed tuned readiness by +0.426 s. Three samples per configuration do not establish a statistically reliable end-to-end gain. The raw transfer improvement does not translate one-for-one into restore latency: startup variation, concurrent CRIU/storage/GPU activity and the serial engine restore path remain.

## CPU evidence

| Configuration | Mean cores used per GMS rank during startup/publication | GMS throttled periods | Main throttled periods |
|---|---:|---:|---:|
| 16 MiB, CPU 1/8 | 0.236 | 0 | 0 |
| 128 MiB, CPU 1/8 | 0.310 | 0 | 0 |
| 128 MiB, CPU 8/16 | 0.313 | 0 | 0 |

GMS counters bracket startup through publication. Main counters cover its lifetime through the second validation request; they are not a restore-only CPU profile. Agent/PageBroker counters bracket each trial including collection. No observed quota throttling does not rule out short CPU bursts, scheduler contention, memory bandwidth or storage contention. Main resources stayed at CPU request 32 / limit 96; this experiment varies GMS CPU, not engine CPU. Higher GMS CPU is not supported as a latency optimization by these results.

## Why +4.88 seconds is not agent startup

The historical create-to-agent field and chart label meant entry into the per-request external restore handler. The daemon and its GPU engine were already running. Every new trial records their running timestamps and Ready status before the measured pod creation. The daemon GPU engine initialized in 8.424 s before the trial series, outside all timers. GMS per-rank startup remains inside the timer.

| Case | Main container started, s | Host running status observed, s | Virtual running status observed, s | Restore operation entered, s |
|---|---:|---:|---:|---:|
| tuned-full-base-1 | 2.574 | 6.449 | 6.454 | 6.343 |
| tuned-full-base-2 | 2.762 | 6.598 | 6.562 | 6.456 |
| tuned-full-base-3 | 1.548 | 5.202 | 5.150 | 5.035 |
| tuned-full-cpu-1 | 2.447 | 6.085 | 6.048 | 5.939 |
| tuned-full-cpu-2 | 1.757 | 5.225 | 5.213 | 5.132 |
| tuned-full-cpu-3 | 1.442 | 4.687 | 4.647 | 4.577 |
| tuned-full-tuned-1 | 1.479 | 4.726 | 4.682 | 4.575 |
| tuned-full-tuned-2 | 0.828 | 4.979 | 4.831 | 4.727 |
| tuned-full-tuned-3 | 1.216 | 5.239 | 5.207 | 5.135 |

Times are relative to the client pod-create request. Running `startedAt` has one-second precision; remote watch receipt includes network latency and cross-clock uncertainty. The handler can appear slightly before this observer receives the status event. These observations do not precisely timestamp kubelet publication, but show seconds of main-start-to-status delay and no comparable additional host-to-virtual propagation delay.

The controller already responds to pod updates and retries runtime discovery on a 50 ms ticker (each lookup has a 1 s timeout). Reducing the general reconcile interval is unlikely to remove this delay. Its runtime lookup filters the virtual pod name/namespace, while containerd labels contain the translated host pod identity. The informer/status path eventually supplies the container ID. A targeted follow-up is resolving the correct host identity, validating the current pod incarnation and container, and using the runtime fast path before status publication. Keep the status fallback. Do not remove identity validation or infer rank from enumeration. No discovery change was introduced during this matrix.

The main and GMS containers are not guaranteed to start simultaneously; each has its own running/process timestamps. Main-first creation and prompt runtime discovery should be investigated independently from GMS transfer tuning. Earlier dispatch may increase useful overlap, but its end-to-end benefit must be measured under contention.

## Measurement and limitations

- All 112 artifact files per trial were opened with O_DIRECT from the PVC. Total GMS payload is 448 GiB (481,036,337,152 bytes); exact matching allocation IDs/sizes and rank/device mapping are verified before weight use. No host-RAM weight staging.
- Sixteen loader workers per rank, two buffers each. Original 16 MiB chunks pin 512 MiB/GPU; tuned 128 MiB chunks pin 4 GiB/GPU (32 GiB/node). Allocation and initialization occur inside the pod timer.
- DRA claim is preallocated, images cached, Snapshot/PageBroker resident, and NFS mounts prepared. CPU checkpoint pages/ghost files receive POSIX_FADV_DONTNEED before each trial; it is advisory. NFS server-side cache state is uncontrolled. This is not a cold-cluster provisioning measurement.
- Same qualified stack, NUMA placement, separate GMS NFS transport and captured default multimem/fused communication configuration in every case. Existing qualified NCCL_IB_DISABLE=1 remains unchanged.
- GMS publication span includes all rank startup/loading skew. CRIU and CUDA are measured agent phase durations; chart placement of CRIU is approximate. Pod readiness includes one coherent generation plus observation overhead.
- Existing raw equal-payload study isolated transfer speed; these full restores include competing CRIU/PageBroker activity. There is no new freshly alternating no-GMS control, so this matrix cannot quantify the isolated GMS-induced slowdown of CRIU/CUDA. Historical agent-only 25.14 s is not an end-to-end pod baseline.

## Evidence

[Interactive Gantt comparison](results/default-config/tuned-study/timelines.html), [all measured values](results/default-config/tuned-study/comparison.json), [resident service proof](results/default-config/tuned-study/agent-ready-before-trials.json), and per-case manifests, logs, O_DIRECT events, CPU counters and pod watches under `results/default-config/tuned-full-*`. Driver scripts are archived as text under `tuned-study/drivers`. One CPU-telemetry setup failure occurred before any restore and is retained separately; it is not a timing sample.

The report should be read alongside [equal-payload transfer results](TRANSFER-COMPARISON.md) and the [DGD startup proposal](DGD-RESTORE.md).

Cleanup was independently verified: experiment pods/claims released, original
agent/operator templates and configuration restored, both private NFS mounts
removed and NUMA balancing restored. The GLM and Qwen snapshots and matching
artifacts remain available. See [cleanup checks](results/default-config/tuned-study/cleanup-verification.json).
Validation: all nine full restored-inference trials passed; Ruff check/format
passed for all 30 benchmark Python scripts.

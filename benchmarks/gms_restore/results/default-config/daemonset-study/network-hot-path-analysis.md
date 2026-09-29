<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Isolated runtime network-discovery experiment

Resolving the engine container and Pod IP from CRI removes the remaining wait for cached Pod IP status after init prestaging. Actual main start → restore handler is **17.79 ms and 19.64 ms** in the two enabled runs, compared with **1,075.58 ms** in the neighboring control. This establishes removal of the startup gap. It does **not** establish a reliable end-to-end improvement: request → workload Ready is **22.603 s and 21.731 s**, versus **22.369 s** for the control.

The execution order is `daemonset-network-1`, `daemonset-preinstalled-3`, `daemonset-network-2`. All three use the same agent revision, **`b209afc332631c5a4c8f747d59dccecd32b4d03e`**, resident GMS, and the same preinstalled CUDA bundle at `/var/lib/schwinns-gms-daemonset-0929/cuda-bundle-e021a725`. The agent was verified Ready before all timers. Both enabled runs set `nvidia.com/gms-prototype-runtime-network-discovery=true`; the control omits it. All three retain `nvidia.com/gms-prototype-runtime-discovery=true`. Earlier init-prestaging and cold/resident results are separate cohorts and are not pooled here.

| Metric | Network 1 | Control: preinstalled 3 | Network 2 |
|---|---:|---:|---:|
| Request → first agent queue insertion (s) | 0.661 | 0.580 | 0.632 |
| Request → actual main start, CRI (s) | 1.780 | 1.865 | 1.695 |
| Request → restore handler (s) | 1.798 | 2.940 | 1.715 |
| Actual main start → handler (ms) | **17.79** | **1,075.58** | **19.64** |
| Request → workload Ready (s) | 22.603 | 22.369 | 21.731 |
| CRIU phase (s) | 7.025 | 6.210 | 6.583 |
| CUDA phase (s) | 8.581 | 7.518 | 8.548 |
| Sum of ten native PREPARE durations (s) | 8.251 | 7.214 | 8.240 |
| Request → first GMS payload read (s) | 0.835 | 0.829 | 0.810 |
| All-rank transfer-completion window (s) | 17.542 | 17.622 | 17.604 |
| Aggregate transfer rate, decimal GB/s | 27.422 | 27.297 | 27.325 |
| Request → all ranks published (s) | 18.432 | 18.583 | 18.430 |
| Request → engine wake gate entered (s) | 20.549 | 19.491 | 19.594 |
| Engine wake gate wait (µs) | 19.79 | 19.55 | 64.37 |

The two enabled runs average 22.167 s to Ready, only 0.202 s below the single control. Their mean handler entry is 1.184 s earlier, including a 1.057 s reduction in main-running → handler wait. CRIU and CUDA both take longer in these enabled samples than in the neighboring control; post-gate inference/readiness also varies. With two enabled runs and one control, these changes cannot distinguish contention from ordinary run-to-run variation or establish a total-latency effect. The evidence supports the specific discovery fix, not a statistically stable speedup for the full restore.

## Mechanism and evidence

The control's first successful `pod_ip_gate` with a cached IP occurs at +2.914 s, after its main container started at +1.865 s. Its handler follows at +2.940 s. Both enabled trials instead log `network_resolution_done` with `source=runtime`, then enter the handler while the original controller Pod snapshot still has no IP. The remote virtual-watch first reports IP at +2.775 s and +2.796 s, approximately 0.977 s and 1.080 s after their handlers. Host-watch observations are +2.788 s and +2.802 s. These watch times are receipt times, not the instant networking became ready.

The enabled resolver requires both opt-ins and the runtime capability. It polls every 50 ms for a unique READY sandbox with the exact native or translated Pod identity and a RUNNING destination container inside it. It then reads sandbox status, rechecks sandbox identity/readiness, validates the primary IP, and returns the container ID and IP together. The controller carries the derived IP in a per-destination restore plan; it does not change informer Pod status. If authoritative same-UID Pod status catches up, the existing status path can still win. Ambiguous identity, invalid IP, unavailable runtime state, or cancellation cannot produce a restore target.

Network 1 resolves in 23 attempts with 45.65 ms cumulative runtime RPC time; network 2 uses 22 attempts and 38.89 ms. Their 1.105 s and 1.053 s polling spans mostly wait for the destination container to start. They are not additional idle time after main start.

Identity and IP checks:

- **Network 1:** resolver container ID equals `cri-starts.json` main ID and the Ready Pod's main container ID. Derived IP `10.0.17.56` matches the Ready Pod and both host/virtual watch streams. The Pod UID matches the operator-created Pod. An additional live sandbox inspection missed cleanup, so this case has no independent `cri-network.json`; its sandbox/IP association relies on the resolver log and validated implementation.
- **Control:** [cri-network.json](../daemonset-preinstalled-3/cri-network.json) independently records RUNNING main, its sandbox ID, READY sandbox status, and IP `10.0.17.36`. Exact virtual UID/name/namespace/kind and translated host name/namespace all match. The host Pod UID is distinct from the virtual UID, as expected.
- **Network 2:** [cri-network.json](../daemonset-network-2/cri-network.json) independently records RUNNING main and READY sandbox, with container → sandbox ID and IP `10.0.17.209` matching `network_resolution_done`. All six identity annotation values match the virtual Pod and host sandbox metadata. The same container ID appears in the collected CRI start record and Ready Pod status.

The extra CRI collector emits only container identity/state/times, sandbox identity/state/network, and the exact six identity annotation keys. It never emits the other sandbox annotations. The collector used for these records is `/tmp/gms-collect-network-proof.py`.

## Scope and checks

All three runs completed coherent inference and have `cpu-main-after.txt`, marking complete collection. Every run has 128 first-read and 128 lane-completion events, eight byte totals summing to **481,036,337,152 bytes (448 GiB)**, and **112 unique payload paths**, with `O_DIRECT` true on every recorded payload open. Payload reads remain inside the request timer. The transfer window is earliest read across all ranks → latest lane completion, not a sum of per-rank rates. All three use resident lanes, so their endpoints share the same buffer-retention behavior. GMS finishes before the engine gate in every case; the engine path remains critical.

Controller milestones use `at_unix_nano`; actual main starts use CRI nanosecond timestamps. Controller logs are filtered by current Pod UID and request-to-Ready window; untagged native PREPARE events are filtered by that time window. Cross-machine request offsets and remote watch observations retain clock/transport uncertainty. The much smaller main-start → handler intervals use node-local clocks and do not depend on rounded Kubernetes `startedAt` values.

The source change is Snapshot commit `b209afc3`: [controller network polling](/home/schwinns/dynamo/worktrees/snapshot-gms-daemonset-20260929/agent/internal/controller/network_discovery.go), [paired CRI resolver](/home/schwinns/dynamo/worktrees/snapshot-gms-daemonset-20260929/agent/internal/runtime/discovery.go), and focused controller/runtime tests. Controller, runtime, and executor package tests passed before deployment. The binary SHA256 is `ef0a146c96c9f8c8bf4705f358c0bb3d132b7fd170d7235f8bb4faaece2b1115`. This remains an opt-in prototype; the normal missing-IP preflight path is unchanged.

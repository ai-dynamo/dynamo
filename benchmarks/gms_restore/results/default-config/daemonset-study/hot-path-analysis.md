<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Primary cold versus resident GMS hot-path analysis

The six primary runs reduce mean DGD request → workload Pod Ready from **25.822 s to 22.781 s**, a **3.042 s (11.8%)** improvement. Resident GMS starts reading sooner; bulk transfer throughput is effectively unchanged. All three resident runs finish publishing weights before the engine reaches its wake gate. The engine restore/wake path becomes critical, with a measurable CRIU slowdown despite the net improvement.

This comparison includes exactly `daemonset-cold-{1,2,3}` and `daemonset-resident-{1,2,3}`, in execution order cold-1, resident-1, resident-2, cold-2, cold-3, resident-3. The first cold run remains included. Later preinstalled-init and network-discovery experiments are separate cohorts.

The timer begins with the client DGD creation request. Ready is workload Pod readiness after coherent inference, not the DGD Ready condition. Both groups use resident Snapshot/PageBroker services and Snapshot revision `d1bb736d31d00db15643759f2e40ebf062288c71`. The resident group additionally starts GMS servers, CUDA contexts, transfer workers, and pinned buffers before the timer. Payload reads and weight allocations remain after the load trigger. Each trial transfers the same **481,036,337,152 bytes (448 GiB)** from **112 unique PVC payload paths**, with `O_DIRECT` asserted on every recorded artifact open. Eight ranks each have 16 lanes and two 128 MiB slots per lane; each GMS container sees one GPU.

| Metric, seconds unless stated | Cold mean (range) | Resident mean (range) |
|---|---:|---:|
| DGD request → workload Ready | 25.822 (24.483–27.656) | 22.781 (21.684–23.391) |
| Request → restore handler | 3.287 (2.257–4.950) | 2.886 (2.470–3.295) |
| Request → first payload read | 5.069 (3.397–6.335) | 0.804 (0.763–0.843) |
| Request → all ranks published | 22.546 (21.518–24.279) | 18.423 (18.099–18.774) |
| Engine wake gate wait | 3.436 (1.523–4.936) | 0.000019 (0.000018–0.000022) |
| CRIU restore phase | 4.600 (3.991–5.261) | 6.473 (6.209–6.634) |
| CUDA restore phase | 8.667 (7.674–10.516) | 8.286 (7.597–8.826) |
| Sum of native PREPARE durations | 8.327 (7.332–10.193) | 7.965 (7.269–8.493) |
| All-rank transfer-completion window | 17.471 (16.361–18.115) | 17.482 (17.326–17.613) |
| Pooled transfer rate, decimal GB/s | 27.534 | 27.517 |

## Transfer accounting and contention

The transfer window starts at the earliest `first_read_start` across all eight ranks and ends at the latest `lane_transfer_complete`. Every trial has 128 start and 128 lane-completion events, and eight `transfer_complete` byte totals summing to 448 GiB. Pooled rate divides three payloads by the sum of the three windows; it does not add concurrent per-rank rates.

| Case | First read after request (s) | Last lane completion after request (s) | Window (s) | Aggregate GB/s | CRIU (s) | CUDA phase (s) |
|---|---:|---:|---:|---:|---:|---:|
| cold-1 | 6.335 | 24.273 | 17.937 | 26.817 | 5.261 | 7.811 |
| cold-2 | 5.474 | 21.835 | 16.361 | 29.402 | 3.991 | 7.674 |
| cold-3 | 3.397 | 21.512 | 18.115 | 26.555 | 4.546 | 10.516 |
| resident-1 | 0.806 | 18.420 | 17.613 | 27.311 | 6.634 | 7.597 |
| resident-2 | 0.843 | 18.349 | 17.506 | 27.479 | 6.575 | 8.436 |
| resident-3 | 0.763 | 18.089 | 17.326 | 27.764 | 6.209 | 8.826 |

There is a measurement asymmetry in [posix_direct.py](../../../posix_direct.py): cold `lane_transfer_complete` follows stream drain, pinned-buffer release, and FD close; resident emits it after draining copies while retaining buffers. The cold endpoint therefore includes cleanup and is not the exact final GPU-copy timestamp. These numbers establish similar application transfer-completion windows, not identical raw storage/H2D bandwidth. A future comparison should add a common event immediately after the final stream drain and before cleanup.

CRIU is slower in every resident run than every cold run: the mean increases **1.873 s (40.7%)**. Earlier GMS reads place more of the 128 concurrent transfer lanes over CRIU. This is consistent with contention, but these six trials do not isolate CPU, memory bandwidth, or storage as its cause. Equal bulk-transfer throughput does not rule out CRIU interference. A controlled concurrency or transfer-start sweep is the next way to identify that tradeoff.

There is no corresponding systematic CUDA slowdown: the CUDA phase mean decreases 0.381 s, and the sum of native PREPARE durations decreases 0.362 s. Each trial contains **ten** PREPARE calls, including auxiliary CUDA processes. `native_prepare_seconds` brackets `cuCheckpointProcessRestore` plus minimal returned-view validation; the enclosing CUDA phase also includes PageBroker scheduling, transfers, and completion. PREPARE is counted once per process; the duplicate value in COMPLETE reports is excluded. The cold-3 CUDA outlier remains in the comparison. These observations do not establish an interposer regression or show that more CPU would help.

## Request-to-handler overhead

CRI `startedAt` timestamps establish that the primary engine containers are not running idle for seconds before Snapshot notices them. Main start → handler is **11.95–19.15 ms cold** and **15.92–35.47 ms resident**. Kubernetes container status and remote watch receipts can arrive much later; bars ending at those observations must not be read as actual container-start latency.

| Boundary | Cold range | Resident range |
|---|---:|---:|
| Request → first agent queue insertion | 0.666–0.838 s | 0.596–0.638 s |
| Request → runtime polling begins | 2.099–3.792 s | 2.039–2.635 s |
| Request → actual main start | 2.242–4.938 s | 2.454–3.259 s |
| Init container created → started | 0.233–1.190 s | 0.258–0.265 s |
| Init container running | 0.126–0.402 s | 0.128–0.339 s |
| Init finished → main created | 0.103–1.572 s | 0.405–0.945 s |

Initial queue insertion → processing is below 0.22 ms in all six cases. The final successful preflight takes 10.05–12.99 ms, the finalizer operation 9.35–16.25 ms, and the InProgress status operation 11.42–21.35 ms. Polling then waits for the main container to exist and run. Faster general reconciliation cannot recover the seconds spent in the observed container lifecycle; actual start → handler already costs only milliseconds in this cohort.

The init lifecycle is a plausible next isolated improvement, particularly the runtime transitions surrounding its short copy command. Its full created → main-created span is 0.462–3.165 s cold and 0.792–1.549 s resident. Removing it does not guarantee that entire span disappears: sandbox/CNI work and runtime scheduling can overlap, and removing one dependency can expose the separate cached-PodIP gate. Preinstalled-init trials must measure the resulting actual main start and handler boundaries before claiming a gain. They do not belong in the primary averages above.

## Evidence and reconstruction

Read each case's `timing.json`, `cri-starts.json`, `publications.json`, `gms-0.txt` through `gms-7.txt`, and `main.txt`. Match timestamped agent milestones to the UID in `operator-created-pod.json`. The resident agent log accumulates multiple trials, so filter native PageBroker events to the current request-to-Ready interval; those events do not carry Pod identity. Use `Restore startup milestone.at_unix_nano` for controller boundaries, and CRI nanosecond strings for actual starts. Use logged phase durations for CRIU/CUDA, not rounded chart placement. The checked cases all have `cpu-main-after.txt`, marking complete collection.

Cross-host wall clocks and remote watch receipt introduce uncertainty in request-relative offsets. Kubernetes creation/started timestamps can have only one-second precision; CRI and same-process durations are the preferred sources. These results support a latency improvement from moving initialization before the request, with remaining engine-path contention to investigate. They do not claim a complete production DGD integration or a faster raw transfer backend.

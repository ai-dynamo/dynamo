<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# DGD-deployed restore: container discovery

CRI sandbox identity discovery reduced mean **DGD request → coherent workload Ready from 27.742 to 24.951 s**, a 2.790 s (10.1%) improvement in three matched pairs. All six comparison restores and both qualification restores passed Berlin generation before readiness and Rayleigh generation afterward.

**Deployment scope:** every new restore workload was created by the Dynamo operator from a DynamoGraphDeployment. The harness validates DGD → DCD → Deployment → ReplicaSet → Pod ownership UIDs. This uses a custom podTemplate with the experimental Snapshot annotation and explicit GMS V1 rank containers. The retained capture serves the SGLang Engine API; native `checkpointRef`, the standard `gpuMemoryService` renderer, Dynamo worker re-registration and frontend routing are not exercised. No compatibility metadata was fabricated.

| Mean of three 32 GiB trials (seconds) | Existing discovery | Identity discovery |
|---|---:|---:|
| DGD request → workload Ready | 27.742 | 24.951 |
| DGD request → child Pod observed | 0.752 | 0.737 |
| DGD request → restore handler | 5.689 | 2.712 |
| CRIU duration | 6.523 | 4.599 |
| CUDA restore phase | 9.313 | 8.940 |
| GMS publication span | 20.576 | 19.103 |
| DGD request → all weights published | 23.659 | 22.062 |
| Engine wait for publication | 0.000 | 3.099 |

[Interactive Gantt](results/default-config/discovery-study/timelines.html) · [all trials and pair differences](results/default-config/discovery-study/comparison.md) · [machine-readable measurements](results/default-config/discovery-study/comparison.json).

## Where the delay was

The restore handler starts 2.978 s earlier on average. The old runtime lookup filters virtual pod names against translated host names and waits for kubelet status instead. The agent was already resident and retrying on a 50 ms ticker; increasing general reconciliation frequency would repeat that mismatched lookup.

The new optional CRI resolver first matches a READY sandbox using the current virtual pod name, namespace and UID, then requires a RUNNING container with the exact requested name in that sandbox. Native pods match their original metadata; vcluster pods require all six translation identity fields, including matching host metadata. Missing/stale identity, ambiguous sandboxes and ambiguous containers fail closed. It preserves the shorter per-attempt deadline and same-UID informer status fallback. The experiment selects it using `nvidia.com/gms-prototype-runtime-discovery: "true"`; old and new paths run in the same resident binary, avoiding daemon restart as an A/B confound.

The Dynamo operator creates the workload promptly in these runs. Child Pod first-observed time includes remote watch delay; server object timestamps have one-second precision and are not precise enough to assign sub-second controller duration. DGD Ready condition observations are recorded separately from the harness workload Ready observation. No multi-second operator reconciliation gate was found, so no speculative operator polling change was made.

Earlier restore exposes the weight-publication dependency: the engine waits 3.099 s on average in the new-path trials. Transfers still overlap CRIU and CUDA restore, and exact publications are checked immediately before weight use. Thus weight availability is now on the critical path in these trials. The measured end-to-end improvement includes changed contention/overlap; it is not an isolated estimate of driver execution saved by faster dispatch.

## Controlled setup

- Same retained default-communication GLM-5.2 TP8 capture and eight B200s; authoritative rank map, allocation IDs/sizes and stable sockets unchanged. One GPU per GMS container; 448 GiB (481,036,337,152 bytes) read from 112 payload files, all verified O_DIRECT.
- Tuned 16 workers × two 128 MiB pinned buffers per rank (4 GiB/GPU), NUMA placement and separate qualified NFS transport; GMS CPU request 1 / limit 8. Engine CPU request 32 / limit 96. GMS startup/allocation/loading are inside the timer.
- Snapshot agent source 8536a8b1 and unchanged qualified CUDA shim/PageBroker. Agent and GPU engine Ready before every DGD timer. The DRA claim is held/preallocated, images cached, PVC mounts prepared, CPU checkpoint pages advised POSIX_FADV_DONTNEED. NFS server cache state is uncontrolled.
- The existing Dynamo operator deployment was used without replacement; its provenance is recorded. It adds its normal metadata/env and owns the generated components. Explicit startup/liveness/readiness probes target the retained app instead of absent Dynamo system-health endpoints.
- Operator-managed /dev/shm defaults to 8 GiB. The first control/runtime pair exposed that template mismatch and remains separately reported as qualification. Setting component sharedMemorySize=32Gi restores the intended capacity. Main pairs use cases 2, 3 and 4 (B–A, A–B, A–B); no outcome-based exclusions.
- Main timing is client DGD request to observed workload Pod Ready, including coherent generation. Pod creationTimestamp and startedAt have one-second precision; raw creation offsets can be slightly negative. API watch observation has network/cross-clock uncertainty. This series is not a cold cluster provisioning benchmark or an end-to-end comparison against historical agent-only 25.14 s measurements.

## Wiring the native Dynamo features next

The current capture cannot simply be labeled as native-compatible: it lacks the required compatibility-version/hash annotations, and its custom entrypoint is not the supported `python -m dynamo.sglang` lifecycle. A native operator-managed SnapshotJob capture must preserve Dynamo runtime re-registration and be verified through its frontend.

The standard GMS renderer currently creates one server using the shared GPU claim and changes the socket directory. Enabling it alongside the eight prototype rank servers would add unrelated wiring rather than preserve the captured contract. The first-class extension needs per-rank V1 server images/tuning, named DRA requests, a PVC artifact-manifest reference, and stable socket aliases. The controller must bind capture rank metadata to snapshot/content identity, resolve destination allocation once, and publish one PodUID-scoped plan for both CUDA restore and GMS. Watch allocation changes and gate weight use on exact publication; do not serialize CRIU behind loader readiness.

For this experiment `run_glm_restore.py --dgd --runtime-discovery` renders the complete custom DGD. The committed `dgd-manifest.json` in each case is the concrete configuration; `operator-created-pod.json` and `ownership.json` prove what the operator actually deployed.

## Validation and reproducibility

Snapshot runtime/controller/executor Go tests passed and the new agent built successfully. Runtime tests cover stale/missing UID, native and virtual identities, wrong sandbox/container, non-running state, ambiguity, backend errors and deadlines. Controller tests cover opt-in, control behavior and refusing name-only fallback. Repeated GPU restores validate the full experimental configuration. Benchmark Python passes Ruff.

The updated complete Snapshot patch is committed as `snapshot-prototype.patch`; source commit and binary SHA256 are recorded in [build-validation.json](results/default-config/discovery-study/build-validation.json). Setup, rotating trial and cleanup drivers are archived under `discovery-study/drivers`. Raw publication, O_DIRECT, CPU, agent, PageBroker, operator and dual-API-watch evidence is retained for every trial.

Cleanup independently verified: experiment DGDs/DCDs, Pods and GPU claims removed; original Snapshot agent/operator templates and configuration restored; Dynamo operator deployment unchanged; private NFS mounts removed and NUMA balancing restored. Retained GLM/Qwen snapshots remain Ready. See [cleanup verification](results/default-config/discovery-study/cleanup-verification.json).

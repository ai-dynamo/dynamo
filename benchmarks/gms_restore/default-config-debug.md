<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Default communication configuration investigation

Date: 2026-09-29. Source: user follow-up. Status: resolved for the tested default configuration; 23 PVC restores validated.
Environment: nscale DRA, eight B200s, pinned Torch 2.11/SGLang 0.5.16 image
and Snapshot d9b6bc72 with prototype external-GMS imports.

Expected: the original evidence workload's multimem logits and FlashInfer fused
allreduce settings work with GMS V1, before and after restore.
Observed previously: stack samples inside symmetric-memory rendezvous; bypassing
the two optimized paths allowed initialization. This alone does not distinguish
autotuning, a shim defect, or a GMS integration issue.

Plan: isolate symmetric-memory rendezvous with and without shim, inspect default
engine startup over time, then validate capture/restore with the default settings.
Reclaim old experiment checkpoints and measure NFS loading separately from RAM.
Use logged timestamps for a Gantt chart; distinguish observed from derived phases.

## Evidence so far

- Retired five old PodSnapshots through the controller (metadata archived), freeing
  about 1.9 TiB. PVC reports 2.8 TiB available. No model-cache files removed.
- Minimal 8-rank Torch symmetric-memory rendezvous passed without the shim, with
  the shim, and with shim + a published GMS V1 model. All returned multicast VAs.
- Full default GLM run loaded weights and published them at 01:57:52 UTC, then
  waited with all eight GPUs at 0% utilization. Collecting native stacks before
  attributing this to autotuning or a defect.
- Prior RAM comparison: CRIU 3.677 s vs PR400 baseline 3.655 s; CUDA phase 8.418 s
  vs 20.467 s; summed native prepare/restore calls 8.105 s vs 10.076 s. These
  are historical observations with communication settings different, not a
  controlled claim about the default configuration.
- Harness now defaults to PVC and rejects non-PVC artifact paths. Both standard
  NIXL staging and prototype direct reads log actual F_GETFL O_DIRECT flags.
- Default startup progressed without communication overrides: autotune completed
  01:58:32–33; graph capture completed 01:59:30; Paris and Rayleigh generation
  succeeded. Native samples showed TVM-FFI JIT compilation/lock waits. The earlier
  blanket claim of communication stalls is not supported by this reproduction.
- After native profiling with output truncated via a pipe, rank 0 disappeared and
  peers reported Gloo connection closure. Container memory.events reported zero
  OOMs. Profiling may have disturbed the process; that causality is not established.
  Repeating without live profiling rather than capturing a damaged process tree.
- User requested overlap and contention comparison. New source explicitly waits
  for /gms/all-ready immediately before resume_memory_occupation. Overlap mode
  runs the validator in a CPU-only sidecar and starts Snapshot as soon as the
  containers run; serialized mode retains the earlier pre-restore gate. Same
  capture, rank map and artifact set will be used for both modes.
- Clean repeat (no live profiling): default graph capture and both coherence
  prompts passed again at 02:06:18 UTC. Now saving exact GMS artifacts to the PVC.
  The default workload still emits the shim's FABRIC-export rejection and uses
  SGLang's automatic POSIX-FD transport fallback; neither multimem logits nor
  FlashInfer fused allreduce is explicitly disabled.
- The previous baseline used the exact same engine image digest. Its conn2 source
  logs also show FABRIC-export rejection followed by POSIX-FD transport. This is
  not introduced by the external-import prototype. Historical evidence path:
  restore-profile-20260925/cases/conn2/source-app.log, lines 556–566.

- Clean default capture `gms-v1-glm-default-0929` reached Ready; content UID
  `859c4057-d1da-45d6-8565-39225861a5be`. CPU ~56 GiB, native GPU 44,275,073,024
  bytes; exact GMS set 481,036,337,152 bytes on PVC `default-capture-2`.
- All six standard NIXL PVC restores passed both coherence requests. Three-pair
  means: serialized 46.083 s pod-to-ready, overlapping 29.562 s. Overlap increases
  preload 1.634 s, CRIU 1.441 s, CUDA phase 0.968 s. This is consistent contention
  evidence, not yet a resource-specific diagnosis. Default multimem/fused settings
  remain enabled. Actual payload FDs all have O_DIRECT.

- Three fused four-worker NUMA pairs completed too. Their ordinary-mount overlap
  mean was 33.474 s, slower than NIXL's 29.562 s; RAM-stage advantages do not
  transfer automatically to PVC. Every trial passed both inference checks.
- Shared qualified NFS mount (same PVC export, O_DIRECT) made NIXL weight loading
  faster: 19.343 s mean over two trials. CRIU rose to 11.875 s and total readiness
  to 34.723 s. One fused eight-worker shared-mount trial reached 27.878 s; sixteen
  workers reached 31.642 s. Testing independent NFS transport before selecting
  and repeating the best configuration. No shared PV mount options were changed.

- Completed 23 PVC/O_DIRECT restores, all passing both inference checks. Best
  repeated setting: fused V1, 16 workers/rank, NUMA affinity, separate qualified
  NFS transport; 27.624 s mean, 27.118–27.895 s range across three trials.
  Its all-rank span averages 22.008 s; per-rank init 0.280 s and loader interval
  20.234 s. CRIU 6.603 s and CUDA phase 8.732 s show that contention remains.
- Ordinary PVC NIXL overlap saves 16.521 s versus its serialized control despite
  phase inflation. Shared qualified-mount NIXL loads faster but increases CRIU to
  11.875 s. The separate-transport experiment reduces that penalty; it does not
  prove that transport queues are the only competing resource.
- The earlier claim of mandatory multimem/fused-allreduce workarounds is withdrawn.
  No new cuInterpose modification was necessary for successful default restores.
  Historical FABRIC/POSIX-FD fallback remains; full distributed Dynamo deployment
  and a freshly alternating no-GMS control remain outside this prototype series.

- Cleanup completed: released experiment pods/claims, restored saved agent/operator
  templates and configuration, restored NUMA balancing, and removed both private
  NFS mounts. Deleted the unused 448 GiB profiled-source weight set. The valid
  GLM default snapshot and Qwen snapshot still report Ready, with their exact
  durable artifacts retained. Verification is archived beside the trial evidence.

## Creation-time dispatch follow-up (2026-09-29)

User asked to minimize blank startup overhead for a real DGD. Representative
prior case: request sent 6.473 s after pod creation; agent started 0.173 s later.
The harness waited for all containers Running, performed host-pod lookup, and
revalidated the claim before sending an already-planned restore request.

- Added `--early-trigger` to put restore intent on pod creation. It requires the
  existing overlap/wake gate, checks claim UID/allocation and rank→UUID mapping
  before creation, and keeps full artifact/publication validation at weight use.
- Initial pre-create validation incorrectly invoked the artifact-reading resolver
  on the orchestration host, where the PVC is not mounted. It failed before pod
  creation. Extracted the named-DRA mapping step; artifact verification still runs
  against the actual PVC inside the pod. Existing rank-plan tests pass.
- First creation-time request with the old agent restored correctly in 48.541 s,
  but agent start was delayed to 32.029 s. The main container started around 2 s.
  The inline runtime resolver holds an initial pod status for up to 30 s and
  looks up virtual pod names in containerd, which stores translated host names.
- Snapshot commit bc42fb31 consumes updated informer status during that loop and
  checks Pod UID to reject IDs from replacement pods. Controller tests pass,
  including late-status progress and replacement-UID rejection. A race-enabled
  invocation could not run with this environment's CGO disabled; no race-test
  success is claimed. The CUDA shim/broker remain unchanged from the capture.
- First fixed-agent early case: agent start 5.102 s; coherent ready 26.370 s.
  Matched control: agent start 6.562 s; coherent ready 26.682 s. CRIU inflation
  offset much of the dispatch gain. Repeating three alternating pairs.
- Current DGD admission already builds restore intent before creation; native
  GMS sidecars have no startup probe. Snapshot+InterPod GMS is currently rejected;
  its Grove startup edges are future integration work. See DGD-RESTORE.md.

- Completed all three alternating fixed-agent pairs. Late/early mean readiness:
  28.076/28.619 s; median 28.082/26.377 s. First-GMS-to-agent mean improved
  4.487→2.629 s. No average end-to-end speedup claim: early-3's first GMS start
  was 9.492 s and total 33.111 s; main-container startup was also late. Keep the
  outlier. Cached images, fast attach and sub-second recorded installer duration
  do not explain the earlier startup gap; exact resource cause remains unknown.
- All eight follow-up restores (two old-agent, six fixed-agent) passed Berlin and
  Rayleigh with the default communication configuration and exact PVC/O_DIRECT
  allocation set. Charts and paired comparison preserve the two agent revisions.
- Cleanup and independent verification complete: no experiment GPU owners,
  original templates/config/operator restored, NUMA balancing 1, private NFS
  mounts absent, retained GLM and Qwen snapshots Ready. DGD integration remains
  a documented proposal; full DGD startup and allocation are not benchmarked.

## Equal-payload transfer comparison (2026-09-29)

User questioned why GMS PVC transfer looked slower than the qualified PageBroker
GPU engine. Existing standard-NIXL serial/overlap pairs measured 1.634 s of
all-rank load inflation; they did not isolate the tuned GMS/PB transfer engines.
The implementations also differ: GMS prototype uses 16 lanes × two 16 MiB slots,
whereas PageBroker preinitializes 32 × 128 MiB CUDA host-NUMA slots and NIXL AIO.

- Allocated an isolated DRA pod on s2877, eight one-GPU containers, same qualified
  PVC transport for both backends. No agent/operator configuration changed.
- The probe runs the unchanged PageBroker TransferBuffers/NixlTransfer source
  through a small C ABI adapter, using the qualified broker's exact NIXL runtime
  and pinned API headers. GMS calls the existing PosixDirect implementation.
- Both consume the exact 448 GiB current captured artifact set into fresh GPU
  allocations. A shared start barrier excludes process/context/destination setup;
  PB ring setup is measured separately, while GMS initializes slots in restore.
  This is a transfer-layer comparison, not a full GMS publication or checkpoint
  restore. GPU bytes are sampled at first/middle/last page of every allocation.
- First PB setup failed because its copied POSIX plugin depended on libaio1t64,
  absent in the workload image. Copied that exact dependency from the broker
  image too; failed setup logs are retained and excluded from timing results.
- Shared-barrier transfers with no CRIU/restore: original GMS 16 MiB chunks
  measured 16.190–17.118 s; qualified PageBroker 128 MiB rings 12.827–13.379 s.
  GMS warm 16 MiB remained 16.486 s; warm 128 MiB reached 12.917–13.035 s,
  and warm 32-lane × 64 MiB reached 12.843–12.955 s. All samples verified.
- CPU telemetry in these warm transfers showed zero quota throttling and roughly
  0.23–0.28 cores/rank for GMS (8-core quota); PB about 0.43 cores/rank.
- User asked about CPU requests and a direct PB→GMS API. Recorded original
  GMS request/limit 1/8 CPUs, main 32/96; script starts staggered by ~1.8 s.
  PageBroker can use V1 RW/allocate/export/commit, but needs scatter/offset-aware
  targets and arbitration with native residual transfers. Design in PAGEBROKER-GMS.md.
- Live resize attempts were rejected by the DRA/vcluster path with HTTP 422
  (only CPU/memory mutable), including a request preserving claim fields. No
  resource change took effect. Recreating only the experiment pod to compare
  request/limit 1/8 against 8/16, retaining the same claim and transfer inputs.

- CPU A–B–B–A complete: 1/8 request/limit mean 16.568 s, 8/16 mean 16.134 s;
  zero transfer-time quota throttling, about 0.25 cores/rank. Small sample and
  variation prevent treating the 0.434 s difference as an established gain.
- Standalone V1 publication 16 MiB: 22.485 and 22.791 s; 128 MiB: 19.692 s.
  The latter actual transfer span is 13.819 s. Startup/allocation work remains;
  this cold standalone setup differs from the previous resident-broker stack.
- Transfer parity is demonstrated for warm larger-buffer configurations, not
  full restore readiness. Direct PB→GMS is a documented integration proposal.
- Retained setup failures and first successful PB sample separately: a code
  refresh raced the second PB launch, and its failed barrier waiters might still
  have been exiting during the first successful sample. Main PB comparison uses
  only the subsequent three unambiguous isolated runs.

- Cleanup verified: transfer pod/claim released, private mount removed, original
  agent/operator templates unchanged, NUMA balancing unchanged, retained GLM and
  Qwen snapshots Ready. No model/checkpoint payload was created or removed here.

## Tuned full restore and CPU-share follow-up (2026-09-29)

User corrected the ambiguous agent-start wording: the node agent and PageBroker
GPU engine must be resident/Ready before the restore timer. Existing +4.875 s
is the per-request external restore entry. Record resident Pod status and broker
ready log before trial clocks, and label the Gantt accordingly.

- Testing a three-condition rotating matrix: 16 MiB + CPU request/limit 1/8;
  128 MiB + 1/8; 128 MiB + 8/16. Three repetitions each, fixed engine resources,
  16 loader lanes, one GPU per rank, exact PVC/O_DIRECT artifacts, overlap and
  creation-time restore request. GMS ring initialization stays inside the pod
  timer. No direct PageBroker/GMS integration is introduced.
- Resident agent is bc42fb31, with unchanged qualified CUDA shim/broker. GPU
  engine initialized its eight contexts and 32 × 128 MiB/GPU rings before tests.
- Lookup already polls at 50 ms; each runtime attempt has a 1 s cap. Containerd
  filters use virtual Pod names while host CRI has translated names. Dual host
  and virtual status watches will locate status-propagation delay.
- First telemetry setup attempted to read privileged agent CPU files at cgroup
  root. Corrected it to resolve /proc/self/cgroup; no restore ran in that attempt.

- All nine full restores completed with Berlin/Rayleigh correctness. Mean
  pod-to-ready: original 27.519 s, tuned 27.053 s, tuned+higher CPU 27.479 s.
  GMS publication span: 21.851 / 20.188 / 19.841 s. No observed GMS or main
  CPU quota throttling. Engine CPU resources stayed fixed; raising GMS CPU
  alone showed no end-to-end benefit. See TUNED-RESTORE.md for scope/caveats.
- Dual watches put host main-start-to-running-status observation at about
  3.25–4.15 s, including timestamp granularity and remote observation latency.
  Host/virtual status arrived close together; external restore entry followed
  status availability promptly. Increasing general reconciliation frequency
  is not supported by this evidence. Correct translated host runtime identity
  is the targeted next experiment; no discovery change entered this matrix.
- Updated Gantt labels distinguish resident daemon from per-request operation,
  and include main startup/status observation. Recorded all manifests, exact
  O_DIRECT paths, per-rank CPU counters, dual pod watches and resident-service
  proof. One pre-restore telemetry setup failure is retained separately.
- Cleanup independently verified: no experiment pods/claims, original agent and
  operator templates/config restored and available, private NFS mounts removed,
  NUMA balancing restored to 1, retained GLM and Qwen snapshots Ready. Ruff check
  and format pass for all 30 benchmark Python scripts; all nine full restores
  are the GPU validation for the new buffer/CPU harness controls.

## Runtime sandbox discovery experiment (2026-09-29)

User requested implementing discovery correction and measuring whether earlier
restore dispatch reduces total time when PVC transfers may remain critical.
Use the existing tuned 128 MiB / CPU request 1 limit 8 composition. Match CRI
sandbox identity by the current virtual Pod UID/name/namespace, then require a
running main container in that sandbox; do not guess translated names. A
prototype annotation will select the new path so alternating A/B trials share
one resident agent/PageBroker instance. Preserve status fallback and test stale
incarnations, ambiguity, unrelated containers and native Pod identity.

- User directed measurement through Dynamo operator. Added DGD podTemplate mode,
  with actual DGD→DCD→Deployment→ReplicaSet→Pod ownership validation and separate
  DGD request/Pod timing. Explicit probes target the captured Engine API sentinel;
  native checkpointRef/GMS feature integration is not asserted for this old
  manually generated capture. Operator deployment remains unchanged.
- New Snapshot agent source 8536a8b1; runtime/controller/executor tests passed.
  Same resident agent supports old path and experimental runtime-discovery opt-in.
  Live CRI sandbox identity fields were checked using an exact six-key allowlist.
- First DGD qualification found operator-managed shared memory defaults to 8 GiB
  despite an explicit dshm Pod volume. Added component sharedMemorySize=32Gi.
  control-1/runtime-1 were already rendered at 8 GiB; retain both as qualification.
  Main comparison is cases 2/3/4 at explicit 32 GiB, six runs total. This exclusion
  is configuration-based and decided before inspecting the matched-run outcomes.
- First control DGD POST returned +0.264 s; child Pod observed +0.727 s. Object
  creation timestamps all occupy one server second, so subtraction may be
  negative due to precision. DGD Ready observed 13 ms after Pod Ready observation.
  No multi-second Dynamo operator gate was observed in that qualification.

- Final matched result: DGD request→workload Ready 27.742→24.951 s (−2.790 s,
  10.1%); handler entry +5.689→+2.712 s. Three same-capacity pairs all improve.
  Eight total restores (six comparison, two qualification) pass both inference
  checks, authoritative rank/publication validation and 112 O_DIRECT payloads.
- Main-runtime engine wake waits 3.139/3.186/2.970 s, passing 85–92 ms after the
  last publication. Weight availability is now on the critical path. CRIU and
  GMS span also improved under changed overlap; do not assign all total-time
  change exclusively to discovery call duration.
- Dynamo operator child-Pod observation averages +0.752/+0.737 s; no multi-second
  reconciliation gate found. Native checkpointRef compatibility and actual
  Dynamo runtime/frontend registration remain a separate capture/integration.
- Independent cleanup checks pass: no test DGD/DCD/Pod/claim, original Snapshot
  stack/settings restored, Dynamo operator deployment unchanged, mounts removed,
  NUMA balancing restored, retained GLM/Qwen snapshots Ready. Ruff passes; targeted
  Go tests pass and the experiment agent was built from source 8536a8b1. Full
  raw evidence, operator-generated manifests and updated Gantt are retained.

## Resident GMS DaemonSet experiment (2026-09-29)

Separate branch `schwinns/gms-daemonset-restore-20260929` isolates the user's
requested interpod experiment. Eight one-GPU V1 server/loader containers warm
contexts and pinned rings before the DGD request; all payload reads remain
PVC/O_DIRECT and start only after a generation-qualified HTTP trigger issued
concurrently with the DGD POST. Fresh DaemonSet generation per trial. Exact
capture IDs, allocation sizes/IDs, socket names and rank-derived UUID mapping
remain enforced. Snapshot agent/PageBroker already Ready before every timer.
Instrument controller preflight and precise CRI start times before attributing
remaining handler delay. Main comparison holds buffer/CPU/NUMA settings fixed.

- Primary three matched pairs: 25.822 s cold → 22.781 s resident (−3.042 s,
  11.8%); all six pass both inference checks. PVC payload transfer windows
  remain ~17.48 s for 448 GiB. Earlier overlap increases CRIU 4.600→6.473 s;
  CUDA phase 8.667→8.286 s does not show a matching slowdown. All resident
  wake gates pass immediately; engine restoration is now critical.
- Exact CRI starts show primary main-running→handler only 12–35 ms; multi-second
  bars ending at Kubernetes status observations are not actual idle time.
- Separate immutable-CUDA-bundle followup verifies all three file SHA256s and removes
  only snapshot-cuda-install. Preinstalled trials 24.469/21.831 s versus nearby
  regular resident 23.391/24.639 s; do not pool with the primary three pairs. Main starts
  around +1.65 s but handler around +2.33 s: cached Pod IP now exposes 0.58–0.79 s
  of real waiting. Testing separate double-opt-in CRI network identity lookup.

- Optional network resolver b209afc33263 binds running container and IP to one
  validated READY sandbox. Same-binary control waits 1.076 s from main start
  to handler; enabled trials wait 17.8/19.6 ms and enter at +1.798/+1.715 s.
  Ready 22.603/21.731 s versus 22.369 s control does not establish an additional
  total-latency gain with n=2 versus n=1. All twelve restores pass independent
  payload, mapping, ownership, restart, CPU/NUMA and inference validation.
- Cleanup verified: original Snapshot stack/config restored; Dynamo operator
  unchanged; experiment DGD/DaemonSet/Pods/claims and private mounts removed;
  NUMA balancing restored; retained GLM/Qwen captures Ready.
- User clarified chart ownership and selected direct PageBroker→GMS as the
  next experiment. Resident GMS already lives in a separate node DaemonSet;
  current transfer is still a V1 loader inside each server process. Charts
  will distinguish ownership explicitly. Audit real PageBroker native import
  and transfer APIs before starting a separate integration worktree.

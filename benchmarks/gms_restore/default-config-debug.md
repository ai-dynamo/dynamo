<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Default communication configuration investigation

Date: 2026-09-29. Source: user follow-up. Status: investigating.
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

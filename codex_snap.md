# Snapshot experiment ledger

Objective: reduce the restore-gate-to-first-token phase below 10 seconds for the
restored model worker. A candidate is promoted only after a reproducible,
production-like restore passes all correctness gates and records that phase below
10 seconds.

## Baseline — 2026-08-17

- PVC cold: CRIU 38.70 s, agent 44.14 s.
- Node-local cache: CRIU 17.07 s, agent 21.53 s.
- tmpfs stable path: CRIU about 10–11 s, agent about 14.5–15.3 s.
- Wake is about 0.39 s and the first request on an awake engine is about 1.64 s.
- The 20.335 GiB checkpoint is almost entirely CRIU pages; `pages-13.img` is 18.093 GiB.

Source: `CHECKPOINT.md` (immutable inputs and retained checkpoint are never modified).

## Experiment E01 — bounded parallel buffered CRIU readers

- Hypothesis: direct 32 MiB chunk reads into restore VMAs can exploit RAM/cache
  bandwidth on buffered/tmpfs restore and reduce CRIU enough to make end-to-end
  TTFT viable below 10 s.
- Evidence before implementation: isolated tmpfs read of the dominant page image
  improved from 11.67 s (one reader) to 1.79 s (eight readers).
- Candidate: `deploy/snapshot/criu-parallel-buffered-restore.patch`, built only
  with the existing bounded 1/2/4/8-reader ladder.
- Correctness gate: every temporary reader is reaped; all task leaders wait at a
  process-tree barrier before any `clone3(set_tid)` restores application threads;
  every reader/reap/barrier error aborts and wakes all waiters.
- Status: **in progress**.  The candidate now fails closed on `ECHILD` and
  uses an all-alive-task process-tree barrier. Static gates and the container
  build pass; privileged end-to-end restore is still the promotion gate.
- Validation 2026-08-17: image
  `dynamo-snapshot-agent:criu-buffered-barrier-8` built with CRIU 4.2 at
  `b47c692`; the privileged multiprocess dump/restore regression passed three
  consecutive times (3.64 s, 3.64 s, 3.60 s). The fixture verifies restored
  heartbeats/checksums and eight readers in at least two task processes.
- Harness correction: use `--leave-running`, then cooperatively stop and reap
  the fixture tree before restore. A regular dump left its original child
  zombies with retained PIDs, producing a false `clone3(set_tid): EEXIST`
  before the candidate reader code could be exercised.
- Live-readiness finding: the retained tmpfs diagnostic pod
  `ghost-kv-l3zero-tmpfs-chunk8-diag` cannot measure this candidate because
  CRIU aborts at 18 ms on `tun: Unable to create tun: No such file or
  directory`, before page restore. Its image also predates this corrected
  patch. A live measurement requires a TUN-compatible target pod and a
  controlled rollout of the new snapshot-agent image; the V2 harness marks
  those cluster mutations as awaiting explicit execution authorization.
- Compatibility correction: the first agent-built CRIU binary required
  `GLIBC_2.38` and could not run in the Jammy vLLM target; it is discarded for
  target use. The same patch was rebuilt from the Jammy CRIU builder and is
  executable in the target image (SHA-256
  `c9d19585de52592c88a588fbeed9078c0b550a57b70538075f58f124dc1b5530`).
- Live result 2026-08-17: the Jammy candidate restored the retained tmpfs
  checkpoint successfully: CRIU 4.870 s, CUDA 6.913 s, agent detection to
  restore-complete 11.983 s. Wake was 391 ms and the streamed completion's
  first byte was 7.8 ms after request. It is a **CRIU promotion** over the
  earlier 10--11 s tmpfs CRIU results, but not an end-to-end promotion: restore
  plus wake is about 12.37 s, above the 10 s objective.
- Next bottleneck: manifest CUDA restore has four PIDs but the agent restores
  them serially. An opt-in bounded 2/4-worker implementation has unit tests;
  default remains serial pending live CUDA validation.

## Experiment E02 — bounded parallel CUDA restore

- Hypothesis: the four independent CUDA checkpoint PIDs can restore with two
  bounded workers, reducing the serial CUDA phase without unlocking any PID
  before every restore operation has succeeded.
- Candidate: `DYN_SNAPSHOT_CUDA_RESTORE_WORKERS=2`, opt-in through the target
  pod; code defaults to one worker and cancels all queued work on first error.
- Live result 2026-08-17: successful tmpfs restore: CRIU 4.473 s, CUDA 4.774 s,
  agent detection to restore-complete 9.439 s. Wake was 401 ms and the valid
  streamed completion emitted its first byte in 97 ms. Phase sum is **9.938 s**.
- Replicate: CRIU 3.803 s, CUDA 6.123 s, agent detection to restore-complete
  10.111 s; wake 388 ms and first byte 92 ms. The phase sum is **10.592 s**.
- Status: **not promoted end-to-end**. Two workers improve the best observed
  CUDA phase but do not control its tail tightly enough for <10 s. Retain it as
  a safe, validated rung; test the bounded four-worker rung next.

## Experiment E03 — four-worker CUDA restore

- Hypothesis: all four CUDA restore PIDs can be restored concurrently on the
  isolated L40S target, retaining the all-restored-before-unlock invariant.
- Live result 2026-08-17: CRIU 3.424 s, CUDA 4.883 s, agent detection to
  restore-complete 8.514 s; successful wake 396 ms; valid streamed canary
  first byte 91 ms. **Measured phase sum: 9.001 s.**
- Safety: restore status `completed`, semantic canary returned `0`, and no
  Xid/OOM/error was found in agent or kernel diagnostics after the run.
- Status: **promoted for the restore phase**. This is the first candidate with
  recorded restore-gate-to-first-token time below 10 s while retaining immutable
  checkpoint, tmpfs cache, and the real wake/canary gate. It does **not** include
  Kubernetes scheduling, placeholder/pod creation, or initialization: a fully
  absent worker to first token remains about 11--13 s in the current path.

## Promotion policy

1. A candidate must pass the privileged multiprocess CRIU regression repeatedly.
2. It must pass a production-like restore from tmpfs, including CUDA/vLLM,
   readiness, wake, canary, PID/TID integrity, and no Xid/OOM.
3. It is promoted only with a recorded restore-gate-to-first-token measurement
   under 10 s. Equal or worse candidates are recorded as discarded with their
   evidence. Pod creation/scheduling is reported separately, never folded into
   this metric.

## Discarded experiments

None yet in this ledger.  Earlier AIO depth 512 experiments did not improve on
the saturated Q=128 PVC baseline and are not candidates for further expansion.

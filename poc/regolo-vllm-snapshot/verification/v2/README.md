# V2-A harness and tournament registry

This directory is separate from the frozen V0.1 protocol and immutable V1
evidence. It contains the offline V2-A harness and the causal registry for V2-A
plus the nine feature lanes. `protocol.draft.json` and a lane digest are not
live-execution authorization; `v2ctl init-run` requires a separate JSON file
containing `{"execution_authorized":true}`.

Execution remains blocked until the user explicitly approves the draft. V2-A
then instruments the existing supported non-GMS path. V2-B requires another
approval and may evaluate exactly one optimization only after V2-A attributes
the bottleneck. GMS is not eligible while Dynamo v1.3.0 documents GMS plus
Snapshot as disabled.

Every run uses the `v2-` prefix, seed `20260814`, new run identifiers, and new
artifact directories. Nothing in V0, I1, or V1 may be modified or overwritten.

## Contents

- `lane.json`: V2-A identity anchored to the retained V1 checksums and immutable
  runtime pins.
- `lanes/`: tournament registry. Pending image/checkpoint identities are explicit;
  they must be replaced with build/runtime digests before a feature lane runs.
- `harness/v2_harness.py`: blinded plan, cooperatively append-only and tamper-evident
  hash-chained ledger, metric parsers, drain proof, safe directory sizing, separate
  diagnosis/promotion gates and candidate-file page-cache advice. Records are bound
  to the frozen lane and sealed schedule digests.
- `harness/v2ctl.py`: offline run initialization, host collection, ledger append
  and verification, gate evaluation, and explicit cold advice.
- `harness/v2_live.py`: authorization-bound sequential runner, V1-compatible
  Pod manifests, UID-preconditioned cleanup and append-only live lifecycle.
- `harness/v2_production.py`: production Kubernetes/CRIU/host/loopback
  collector plus bounded checkpoint cache advice and storage characterization.
- `tests/`: stdlib tests requiring no GPU, Kubernetes, network, or credentials.

Initialize a new V2-A artifact directory only after authorization has been
provided out of band:

```bash
python3 harness/v2ctl.py init-run \
  --lane lane.json --authorization /secure/path/v2-a-authorization.json \
  --output /new/artifact/path/v2-a-20260814
```

Collect host-side evidence using explicit paths. This command performs no
Kubernetes or network operation:

```bash
python3 harness/v2ctl.py collect-host \
  --meminfo /proc/meminfo \
  --psi-cpu /proc/pressure/cpu --psi-io /proc/pressure/io \
  --psi-memory /proc/pressure/memory \
  --io-stat /sys/fs/cgroup/io.stat --diskstats /proc/diskstats \
  --size checkpoint=/absolute/checkpoint/path
```

Cold simulation is file-scoped and opt-in. The only cache operation is
`POSIX_FADV_DONTNEED`; global `drop_caches`, remounts, writes, and deletion are
not implemented:

```bash
python3 harness/v2ctl.py cold-advise \
  --allow-root /absolute/checkpoint/path \
  --file /absolute/checkpoint/path/pages-1.img
```

The harness does not create feature branches/worktrees, images, or checkpoints.
Those identities remain `pending:*` until V2-A is executed, analyzed, and sealed;
this preserves the plan's prohibition on feature work before the comparative
foundation report.

`gate` evaluates V2-A evidence completeness by default. `gate --optimized` adds
the first-token, CRIU, paired GPU-memory and throughput promotion thresholds;
the diagnostic baseline is not required to meet future optimization targets.

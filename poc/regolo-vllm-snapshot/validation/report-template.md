# Dynamo Snapshot PoC validation report

Status: **NOT EXECUTED**

## Answer

Decision: pending (`Go`, `Optimize`, `No-Go`, or `Protocol invalid`).

## Frozen protocol

- Verification version: V0.1
- Dynamo: v1.3.0, commit `8ce9e22f11576402102ea9d8b8e46233f5430a0d`
- V0 checksum manifest: attach `verification/v0/SHA256SUMS`
- Run-plan checksum: pending
- Compatibility hash: pending

## Environment

Attach the immutable V0 and V1 inventory JSON documents. State the Regolo
source digest, candidate digest, model revision, original command/args, complete
PodSpec, L40S UUID, driver, CRI runtime, StorageClass, PVC properties, probe,
mounts and observed cache conditions.

## Results

| Metric | Cold | Restore | Requirement |
| --- | ---: | ---: | --- |
| Eligible runs | pending | pending | 10 + 10 |
| Median Pod to Ready | pending | pending | reported |
| Median Pod to HTTP 200 | pending | pending | reported |
| Median Pod to first token | pending | pending | restore >=3x faster for Go |
| Valid responses | pending | pending | 20/20 |
| GPU memory | pending | pending | each pair within +/-5% |
| Restore success | n/a | pending | 10/10 |

- Checkpoint duration: pending
- Checkpoint size: pending
- Break-even restores: pending

Attach `blinded-summary.json` and its checksum before attaching the unblinding
key and `final-report.json`.

## Failures, exclusions, and deviations

List every retained failure. For an exclusion, link its raw Kubernetes event
evidence and the repeated complete paired block. Any departure from V0.1 makes
the outcome `Protocol invalid` unless it was approved and frozen as a new V0.

## Security and retention

Confirm the checkpoint was taken before any inference prompt, the KV cache was
empty, credentials were temporary, storage encryption was verified, no secret
was embedded in image/checkpoint, and the approved retention/deletion action.

# Dynamo Snapshot V1 restore diagnosis and V2 proposal

Date: 2026-08-13 UTC
Status: diagnosis complete; V2 draft awaits explicit approval

## Answer

The 36.267 s restore median and 143.663 s tail are dominated by CRIU restoring
the checkpoint's CPU-side page images. The slow run is not explained by
scheduling, admission, CUDA restore, readiness, or first inference.

For `v1-04-01`, first-token latency exceeds the restore median by 107.396 s.
CRIU exceeds its own median by 106.567 s, accounting for 99.23% of that excess.
CRIU and first-token latency have Pearson `r=0.999887` across all ten restores.

This identifies the responsible phase with high confidence. Existing evidence
supports, but does not yet causally prove, that page-cache/storage behavior is
the source of CRIU's variability. V2-A is designed to measure that distinction.

## Measured phase breakdown

All values were independently reconstructed from immutable V1 results and raw
Kubernetes events plus the retained snapshot-agent log. No V1 artifact was
modified and no run was excluded.

| Phase | Min | Median | nearest-rank p95 / max |
| --- | ---: | ---: | ---: |
| Pod -> Scheduled | 0.027 s | 0.120 s | 1.001 s |
| Pod -> agent restore start | 1.693 s | 2.403 s | 3.276 s |
| CRIU restore | 20.281 s | 25.323 s | 131.890 s |
| CUDA restore | 6.961 s | 7.530 s | 8.028 s |
| Total measured restore | 27.968 s | 32.424 s | 139.830 s |
| Restore summary -> first token | 1.033 s | 1.483 s | 1.518 s |
| Pod -> first token | 31.205 s | 36.267 s | 143.663 s |

The checkpoint contains 43,837,075,456 bytes of CRIU `pages-*` data out of
43,886,233,878 bytes total. One file, `pages-12.img`, contains 42,378,522,624
bytes. The manifest has `lazyPages: false`, so those pages are materialized
before CUDA restore. The pinned agent uses upstream CRIU 4.2 at commit prefix
`b47c692`.

Checkpoint bytes divided by CRIU duration yield an effective processing rate
of 0.333-2.164 GB/s, with median 1.733 GB/s. This is not a pure disk benchmark:
CRIU work is included. Its 6.5x range nevertheless matches the observed latency
range, while CUDA varies by only 1.07 s. The first four CRIU restores took
55.921-131.890 s; the last six took 20.281-25.347 s. That temporal shift is
consistent with cache warming or changing backing-storage service time, but
historical disk/page-cache telemetry was not collected in V1.

The backing device is exposed as a rotational `QEMU HARDDISK`, with LUKS2 and
ext4 layered above it. No kernel OOM, block-device, ext4, dm-crypt, or NVIDIA
warning coincided with the V1 window. Kubernetes reported the slow Pod Running
in 2.718 s, confirming that its additional two minutes occurred after normal
Pod startup.

## Ordered bottlenecks

1. CRIU must restore a 43.84 GB page image using the supported upstream path.
2. CRIU effective processing rate is non-stationary; storage/page-cache state is
   the leading explanation for the tail but needs per-run I/O evidence.
3. CUDA restore contributes a stable median 7.530 s.
4. Pod creation through agent restore start contributes a median 2.403 s.
5. Restore completion through the first valid token contributes a stable median
   1.483 s.

The best observed supported-path run is already 31.205 s. Reaching 15 s or less
requires removing most of the CRIU page-image cost, not tuning probes or
scheduling. NVIDIA's current technical description independently identifies
upstream CRIU's serial `preadv` memory restore as a large-model bottleneck and
describes parallel memfd restore, native AIO, KV-cache release, and GMS as the
paths used in faster prototypes. It also states that the CRIU optimizations are
not yet shipped in Dynamo Snapshot. Current v1.3.0 documentation says GMS plus
Snapshot is disabled, so the safety gate must not be overridden.

Sources:

- <https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/>
- <https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/advanced-platform/snapshot>

## V2 proposal

The machine-readable proposal is `../verification/v2/protocol.draft.json`.

V2-A keeps the V1 model, revision, command, Dynamo version, GPU, driver, node,
image, cache, encrypted PVC, security controls, pairing, and blinding. It uses
a new seed and 20 paired blocks so nearest-rank p95 is not merely the maximum.
It adds phase-complete agent timestamps, checkpoint read bytes/throughput,
diskstats, I/O pressure, CPU, available memory, and page-cache measurements.
A bounded read-only storage characterization compares first, repeated, and
direct sequential reads without global `drop_caches`.

V2-A determines causality and a stable supported-path baseline; it changes no
restore architecture. V2-B then evaluates exactly one approved optimization.
Its Go gate is 20/20 successful restores, 40/40 valid responses, no more than 5%
paired GPU-memory drift, first-token median <=15 s, nearest-rank p95 <=25 s,
maximum <=40 s, and CRIU p95 <=12 s. Failures and tail observations are always
retained.

These targets are deliberately absolute. They represent movement toward the
roughly 10 s Dynamo reference while allowing the measured 2.4 s Pod/agent and
1.5 s post-restore overhead. They cannot be claimed by merely prewarming the
page cache.

## Security, performance, and rollback

- Service-account token mounting remains disabled; storage remains encrypted;
  no customer traffic or customer prompt is allowed.
- V2 uses new image identities, compatibility hashes, run IDs, and `v2-*`
  directories wherever a candidate changes.
- V0, I1, and V1 remain immutable. Namespace, PVC, checkpoint, candidate image,
  and keys remain retained until cleanup is explicitly approved.
- Measurement aborts on node pressure, GPU Xid, I/O error, correctness failure,
  or secret-audit failure.
- GMS, optimized CRIU, storage changes, driver changes, and feature-gate changes
  each require separate explicit approval. A disabled safety gate is never an
  acceptable experiment toggle.

## Reproducibility

The derived evidence is `evidence/v1-restore-phase-analysis.json`
with SHA-256
`e6dec04dcf1b0cf484889587126dec50f11582f39527d9c2c81fcf716b91a36d`.
The analyzer validates exactly one start and timing summary for each of all ten
restore runs and fails on missing or duplicate evidence.

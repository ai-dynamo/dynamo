# GMS Multi-GPU Transfer Benchmark Worklog

Date: 2026-05-15

## Context

Goal: clarify whether the PVC/VAST default NIXL backend is genuinely comparable
to the Sharded SSD prototype once concurrent multi-GPU loads/stores are measured.
The primary decision metric is restore/load throughput because the expected
workflow checkpoints once and restores many times.

Important measurement rule from review: include backend creation, agent/backend
setup, memory registration, transfer initialization, data movement, and final CUDA
synchronization in the timed restore/store region.

## Runtime Setup

- Kubernetes context: `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster`
- Namespace: `schwinns`
- Node: `cluster-0967a26d-pool-14bee067-prctr-s2877`
- Pod: `gms-transfer-bench-s2877`
- Runtime image:
  `nvcr.io/nvidian/dynamo-dev/schwinns:dynamo-vllm-placeholder-quiesce-sm80sm100-tp2-dynsnapshot-filepg-rt4-ncclsm80-nosplit-nsrestore-20260514T204156Z`
- Source overlay: copied current local worktree from
  `/tmp/dynamo-gms-transfer-mainbase/lib/gpu_memory_service` to
  `/tmp/dynamo-src/lib/gpu_memory_service` in the pod.
- Benchmarks run with `PYTHONPATH=/tmp/dynamo-src/lib`.
- The 8-GPU pod sees eight NVIDIA B200 GPUs.

Mounts:

- PVC/VAST: `/checkpoints`, NFSv3,
  `rsize=1048576,wsize=1048576,nconnect=16`
- Local NVMe: `/mnt/nvme2` ... `/mnt/nvme9`, ext4

Cleanup performed for 72 GiB/GPU tests:

- Removed generated benchmark output directories:
  `/checkpoints/gms-transfer-concurrent-store` and `/mnt/nvme*/sharded-d*`.
- User allowed deleting older checkpoints. Removed non-`gms` top-level
  directories under `/checkpoints`, leaving `/checkpoints/gms`.
- Post-cleanup VAST free space: ~839 GiB.
- After regenerating the 8x72 GiB VAST input dataset, current VAST free space is
  ~263 GiB. Current top-level VAST dirs are `/checkpoints/gms` and
  `/checkpoints/gms-synth-8gpu-72g`; the VAST store-output directory is no
  longer present.
- For benchmark isolation, scaled graph worker components to zero:
  - `hwoo-snapshot/mx-gms-cold-qwen3-06b` component `VllmDecodeWorker`
  - `hwoo-snapshot/mx-gms-snapshot-qwen3-06b` component `VllmDecodeWorker`
  - `dynamo-qa-ci/sglang-agg-gms-rc1` component `decode`
  These were DynamoGraphDeployment components; scaling child Deployments/DCDs was
  not sufficient because the graph controller reconciled them back to 1.
- After the benchmark run, restored those three graph worker components to
  replicas=1 and verified the DynamoGraphDeployments reached Ready again.
- Deleted benchmark pod `schwinns/gms-transfer-bench-s2877` after measurements
  to release the 8 GPUs on s2877.
- Recreated the same benchmark pod for the direct-I/O rerun, repeated the
  temporary scale-down/restore cycle for those graph workers, verified all three
  DynamoGraphDeployments reached Ready again, and deleted the benchmark pod
  again after the ablation.

## Harness

Created `/tmp/gms_multi_gpu_bench.py`.

The harness launches one Python worker per GPU and synchronizes workers with a
barrier. Reported `critical_path_elapsed_s` is the maximum timed worker section,
and aggregate GiB/s is total bytes divided by that slowest worker time.

For load workers, the timed section includes:

- `create_transfer_backend`
- backend agent/backend creation
- registrations and transfer setup performed by the backend
- restore data movement
- `torch.cuda.synchronize(device)`
- session/backend close

For store workers, the timed section includes:

- GPU source tensor to pinned host staging
- file writes
- `fsync`
- pinned buffer cleanup
- final CUDA synchronization

`barrier_to_exit_elapsed_s` is also reported, but it includes process teardown and
is not the primary transfer number.

## Direct-I/O Rule

All backend comparisons should bypass the client page cache where the backend
allows it.

- Sharded SSD `preadv` opens files with `O_DIRECT`.
- NIXL GDS bypasses the CPU page cache by design.
- NIXL POSIX does not appear to expose a backend parameter for direct I/O. The
  POSIX backend queues I/O against the file descriptor embedded in the FILE
  descriptor, and NIXLBench enables direct I/O by opening storage files with
  `O_DIRECT` before registration. GMS therefore needs to open NIXL POSIX FILE
  descriptors with `O_DIRECT`.
- Verified in the benchmark pod that both `/checkpoints` VAST/NFS files and
  `/mnt/nvme*` local SSD files accept `O_RDONLY|O_DIRECT`; `F_GETFL` showed
  `O_DIRECT` set for both.
- After this check, patched the benchmark overlay so NIXL POSIX uses
  `O_DIRECT` and refuses to silently fall back to buffered reads.

Server-side VAST caching may still exist outside the pod's control. The claim
here is client page-cache bypass through direct file descriptors.

## Earlier Baseline, Before NIXL Direct Open

PVC/VAST default NIXL load:

- 64 GiB, 8 workers: 8.47 / 18.40 GiB/s
- 64 GiB, 16 workers: 22.53 / 24.01 GiB/s
- 64 GiB, 32 workers: 28.56 / 29.51 GiB/s
- 128 GiB, 32 workers: 16.52 GiB/s

Sharded SSD `preadv/O_DIRECT` load:

- 8 GiB smoke, 8 workers: 11.35 GiB/s
- 64 GiB, 8 workers: 30.24 / 30.89 GiB/s

Sharded SSD NIXL POSIX experiments:

- Current image `nixl 0.10.1`, naive per-chunk registration:
  10.62 / 16.47 GiB/s
- `nixl 1.1.0` overlay, matching plugin path, naive per-chunk registration:
  9.43 / 16.13 GiB/s
- `nixl 0.10.1`, prepped descriptor-list probe, balanced 8 roots:
  8.29 GiB/s
- `nixl 1.1.0`, prepped descriptor-list probe, balanced 8 roots:
  13.77 GiB/s

PVC/VAST filesystem read/write microbench:

- 64 GiB write, 16 workers: 5.32 GiB/s
- 64 GiB read, 16 workers: 12.65 GiB/s first read, 34.89 GiB/s second read

Interpretation: these PVC/VAST NIXL POSIX numbers were measured before GMS forced
`O_DIRECT` on the NIXL FILE descriptors. They are useful history but should not
be used for backend decisions. The high 64 GiB VAST numbers were likely
page-cache or warm-cache assisted.

## Multi-GPU Load Results

Each run used 16 GiB per GPU, unique source byte ranges where possible.

PVC/VAST default NIXL:

| GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---:|---:|---:|---:|---|
| 1 | 16 | 1.235 | 12.95 | 5 files |
| 2 | 32 | 1.322 | 24.21 | scales roughly linearly |
| 4 | 64 | 1.481 | 43.23 | strong scaling |
| 8 | 128 | 4.196 | 30.51 | contention appears; GPUs 4-7 much slower |

Sharded SSD `preadv/O_DIRECT`:

| GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---:|---:|---:|---:|---|
| 1 | 16 | 0.754 | 21.21 | faster than PVC |
| 2 | 32 | 1.300 | 24.62 | similar to PVC |
| 4 | 64 | 1.965 | 32.57 | slower than 4-GPU PVC in this run |
| 8 | 128 | 3.412 | 37.51 | faster than 8-GPU PVC |

Production-sized synthetic 8x72 GiB load:

| Backend | GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---:|---|
| PVC/VAST default NIXL | 8 | 576 | 28.009 | 20.56 | synthetic 18x4 GiB shards/device, first run |
| PVC/VAST default NIXL | 8 | 576 | 32.384 | 17.79 | synthetic 18x4 GiB shards/device, repeat run |
| Sharded SSD `preadv/O_DIRECT` | 8 | 576 | 12.218 | 47.14 | synthetic 18x4 GiB shards/device, first run |
| Sharded SSD `preadv/O_DIRECT` | 8 | 576 | 12.170 | 47.33 | synthetic 18x4 GiB shards/device, repeat run |

Raw output snippets:

- PVC/VAST 8-GPU load:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=16.00 workers_per_gpu=8 total_gib=128.00 critical_elapsed=4.196s bw=30.51GiB/s barrier_to_exit_elapsed=5.394s`
- Sharded SSD 8-GPU load:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=16.00 workers_per_gpu=8 total_gib=128.00 critical_elapsed=3.412s bw=37.51GiB/s barrier_to_exit_elapsed=4.741s`
- PVC/VAST 8x72 GiB synthetic load:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=28.009s bw=20.56GiB/s barrier_to_exit_elapsed=29.804s`
- PVC/VAST 8x72 GiB synthetic load, repeat:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=32.384s bw=17.79GiB/s barrier_to_exit_elapsed=34.175s`
- Sharded SSD 8x72 GiB synthetic load:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=12.218s bw=47.14GiB/s barrier_to_exit_elapsed=14.235s`
- Sharded SSD 8x72 GiB synthetic load, repeat:
  `SUMMARY multi-gpu role=load devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=12.170s bw=47.33GiB/s barrier_to_exit_elapsed=14.206s`

Interpretation: the Sharded SSD `preadv/O_DIRECT` numbers remain valid, but the
PVC/VAST NIXL POSIX rows above are superseded by the direct-open rerun below.

## Direct-I/O Load Rerun

After forcing NIXL POSIX FILE descriptors to open with `O_DIRECT`, remeasured the
NIXL POSIX cases and the worker-count ablation on the current runtime NIXL
version (`nixl`/`nixl-cu13` 0.10.1). The older NIXL 1.1.0 overlay probes above
were not used for the final recommendation.

PVC/VAST NIXL POSIX single-GPU load, client page cache bypassed:

| Case | Workers/GPU | Critical Path s | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---|
| 64 GiB | 8 | 4.670 | 13.70 | 16 source files |
| 64 GiB | 16 | 3.940 | 16.24 | best 64 GiB point |
| 64 GiB | 32 | 3.983 | 16.07 | no gain past 16 sources |
| 128 GiB | 32 | 6.939 | 18.45 | 32 source files |

PVC/VAST NIXL POSIX small multi-GPU load, 16 GiB/GPU, 8 workers/GPU:

| GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---:|---:|---:|---:|---|
| 1 | 16 | 1.308 | 12.23 | 4 files/GPU |
| 2 | 32 | 2.217 | 14.43 | |
| 4 | 64 | 3.381 | 18.93 | old pre-direct 4-GPU win disappears |
| 8 | 128 | 5.908 | 21.66 | |

Production-sized 8x72 GiB load worker-count ablation:

| Backend | Workers/GPU | Critical Path s | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---|
| PVC/VAST NIXL POSIX `O_DIRECT` | 1 | 36.211 | 15.91 | under-parallelized |
| PVC/VAST NIXL POSIX `O_DIRECT` | 2 | 29.284 | 19.67 | |
| PVC/VAST NIXL POSIX `O_DIRECT` | 4 | 26.612 | 21.64 | reaches plateau |
| PVC/VAST NIXL POSIX `O_DIRECT` | 8 | 27.240 | 21.15 | no gain |
| PVC/VAST NIXL POSIX `O_DIRECT` | 12 | 26.339 | 21.87 | best PVC/VAST point |
| PVC/VAST NIXL POSIX `O_DIRECT` | 18 | 27.026 | 21.31 | no gain |
| Sharded SSD `preadv/O_DIRECT` | 1 | 24.787 | 23.24 | under-parallelized |
| Sharded SSD `preadv/O_DIRECT` | 2 | 12.534 | 45.96 | near plateau |
| Sharded SSD `preadv/O_DIRECT` | 4 | 12.382 | 46.52 | |
| Sharded SSD `preadv/O_DIRECT` | 8 | 12.238 | 47.07 | best measured point |
| Sharded SSD `preadv/O_DIRECT` | 12 | 12.267 | 46.96 | no gain |
| Sharded SSD `preadv/O_DIRECT` | 18 | 12.170 | 47.33 | previous repeat; same path |

Sharded SSD with NIXL POSIX FILE reader, direct file descriptors:

| Case | Workers/GPU | Critical Path s | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---|
| 1 GPU, 64 GiB | 8 | 1.841 | 34.76 | faster than earlier buffered/naive NIXL POSIX probes |
| 8 GPUs, 72 GiB/GPU | 2 | 12.703 | 45.34 | near `preadv/O_DIRECT` |
| 8 GPUs, 72 GiB/GPU | 8 | 12.344 | 46.66 | essentially tied with `preadv/O_DIRECT` |

Updated interpretation:

- The prior PVC/VAST NIXL numbers were materially affected by not forcing direct
  file descriptors. Once NIXL POSIX uses `O_DIRECT`, PVC/VAST no longer has a
  small 4-GPU win; it reaches only 18.93 GiB/s at 4x16 GiB and 21.66 GiB/s at
  8x16 GiB.
- For the production-sized 8x72 GiB restore, PVC/VAST NIXL POSIX needs at least
  4 workers/GPU, then plateaus around 21-22 GiB/s. More workers do not change the
  conclusion.
- Sharded SSD needs at least 2 workers/GPU to hit its plateau, then stays around
  46-47 GiB/s. More than 2-4 workers/GPU is not useful on this dataset.
- With direct file descriptors, NIXL POSIX is no longer a performance problem
  for the Sharded SSD path. Local SSD via NIXL POSIX is essentially tied with
  local SSD via `preadv/O_DIRECT` at production scale.

Updated load-focused recommendation:

## 2026-05-18 NIXL Common Executor Refactor Check

Purpose: verify that factoring shared NIXL transfer-handle lifecycle code and
making the `nixl-gds` backend use a bounded in-flight transfer window did not
regress the default NIXL POSIX staging path or the Sharded SSD NIXL POSIX
staging path.

Code overlay:

- Local worktree: `/private/tmp/dynamo-gms-transfer-mainbase`
- Overlay copied into pod: `/tmp/dynamo-src/lib/gpu_memory_service`
- Benchmark pod: `schwinns/gms-transfer-bench-s2877`
- Runtime image unchanged from prior run:
  `nvcr.io/nvidian/dynamo-dev/schwinns:dynamo-vllm-placeholder-quiesce-sm80sm100-tp2-dynsnapshot-filepg-rt4-ncclsm80-nosplit-nsrestore-20260514T204156Z`
- Timed region includes backend creation, NIXL agent/backend creation,
  registrations, transfer setup, data movement, CUDA synchronization, and
  session/backend close. GPU destination allocation happens before the timed
  region, matching the earlier harness shape.
- Temporary isolation: scaled these worker components to zero before running and
  restored them after:
  - `hwoo-snapshot/mx-gms-cold-qwen3-06b` component `VllmDecodeWorker`
  - `hwoo-snapshot/mx-gms-snapshot-qwen3-06b` component `VllmDecodeWorker`
  - `dynamo-qa-ci/sglang-agg-gms-rc1` component `decode`

Regression check results, 8 GPUs, 72 GiB/GPU, 18x4 GiB files/device:

| Backend | Workers/GPU | Critical Path s | Aggregate GiB/s | Previous Comparable Band | Interpretation |
|---|---:|---:|---:|---:|---|
| Sharded SSD NIXL POSIX `O_DIRECT` | 8 | 12.352 | 46.63 | 46.66-47.33 | No meaningful regression |
| PVC/VAST NIXL POSIX `O_DIRECT` | 8 | 25.888 | 22.25 | ~21-22 | No regression; slightly above prior 8-worker point |

Single-GPU smoke:

- Sharded SSD NIXL POSIX `O_DIRECT`, 1 GPU, 72 GiB, 8 workers:
  2.579s critical path, 27.91 GiB/s. This was only a setup smoke and is not
  used for the multi-GPU recommendation.

GDS smoke:

- Tried `nixl-gds` on the local SSD synthetic dataset, 1 GPU, 72 GiB,
  8 max in-flight transfers.
- The run failed before transfer start while registering the first FILE memory
  descriptor:
  `GDS_MT: failed to create file handle: GDS_MT: file register error: error=5027`
  followed by `nixlBackendError: NIXL_ERR_BACKEND`.
- This does not measure the new bounded-window logic. It means this benchmark
  pod/filesystem setup did not successfully support the NIXL GDS file path.

Refactor-specific interpretation:

- The POSIX staging hot loop was not abstracted behind a per-chunk callback, and
  the production-sized Sharded SSD/VAST numbers stayed in the prior bands.
- The common NIXL transfer window is therefore safe for handle-level routes
  such as GDS and future UCX P2P without putting measurable overhead into the
  POSIX staging path.

- Default PVC/VAST NIXL POSIX wins on operational simplicity when the checkpoint
  already lives on PVC/VAST and restore latency is not the primary goal.
- Sharded SSD wins for performance on local-NVMe nodes, including single-GPU
  direct reads and the target 8-GPU 72 GiB/GPU restore. At production scale it is
  about 2.1x-2.2x faster than PVC/VAST NIXL POSIX after both paths bypass the
  client page cache.
- For Sharded SSD, either `preadv/O_DIRECT` or NIXL POSIX with enforced direct
  file descriptors is performance-viable. The choice should be made on code
  cleanliness and backend unification, not on measured throughput.
- Suggested concurrency defaults from this ablation: PVC/VAST NIXL POSIX should
  use at least 4 workers/GPU; Sharded SSD should use at least 2 workers/GPU, with
  4 workers/GPU as a conservative default.

## Multi-GPU Store Results

Each run wrote 16 GiB per GPU using GPU-to-pinned-host staging, 8 writer threads
per GPU, and `fsync`.

PVC/VAST store:

| GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---:|---:|---:|---:|---|
| 1 | 16 | 2.899 | 5.52 | |
| 2 | 32 | 5.133 | 6.23 | total improves slightly |
| 4 | 64 | 8.806 | 7.27 | several GPUs around ~1.8 GiB/s |
| 8 | 128 | 18.168 | 7.05 | most GPUs around ~0.88 GiB/s |

Sharded SSD store, GPU-to-pinned-host staging, `O_DIRECT`, 8 writer threads per
GPU:

| GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---:|---:|---:|---:|---|
| 1 | 16 | 0.869 | 18.41 | |
| 2 | 32 | 1.415 | 22.61 | |
| 4 | 64 | 2.820 | 22.69 | |
| 8 | 128 | 5.982 | 21.40 | one GPU was much faster; others ~2.7 GiB/s |

Production-sized synthetic 8x72 GiB store:

| Backend | GPUs | Total GiB | Critical Path s | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---:|---|
| PVC/VAST | 8 | 576 | 97.892 | 5.88 | failed once with quota while VAST input dataset was still present; valid rerun after cleanup |
| Sharded SSD `O_DIRECT` | 8 | 576 | 21.464 | 26.84 | GPU-to-pinned-host staging, 18 writers/GPU |

Raw output snippets:

- PVC/VAST 8-GPU store:
  `SUMMARY multi-gpu role=store devices=8 per_gpu_gib=16.00 workers_per_gpu=8 total_gib=128.00 critical_elapsed=18.168s bw=7.05GiB/s barrier_to_exit_elapsed=21.534s`
- Sharded SSD 8-GPU store:
  `SUMMARY multi-gpu role=store devices=8 per_gpu_gib=16.00 workers_per_gpu=8 total_gib=128.00 critical_elapsed=5.982s bw=21.40GiB/s barrier_to_exit_elapsed=10.346s`
- PVC/VAST 8x72 GiB synthetic store:
  `SUMMARY multi-gpu role=store devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=97.892s bw=5.88GiB/s barrier_to_exit_elapsed=99.544s`
- Sharded SSD 8x72 GiB synthetic store:
  `SUMMARY multi-gpu role=store devices=8 per_gpu_gib=72.00 workers_per_gpu=18 total_gib=576.00 critical_elapsed=21.464s bw=26.84GiB/s barrier_to_exit_elapsed=25.911s`

Interpretation:

- Store also strongly separates the backends, but it is secondary to restore
  latency for the expected checkpoint-once/restore-many workflow.
- PVC/VAST write throughput saturates around 7 GiB/s total in this setup, and
  per-GPU write rate collapses under 8 concurrent writers.
- Sharded SSD write throughput saturates around 21-23 GiB/s total in this setup,
  roughly 3x the 8-GPU PVC/VAST store number.
- For product prioritization, load speed matters more than store speed because
  checkpointing is expected to happen once while restore happens many times. The
  store numbers are useful supporting context, not the primary backend decision.
  They point in the same direction as load, but they should not be weighted as
  heavily as the 8x72 GiB restore measurements.

## Data And Cleanup Notes

- Synthetic 8x72 GiB dataset generation:
  - VAST path: `/checkpoints/gms-synth-8gpu-72g`
  - Sharded SSD path: `/mnt/nvme*/gms-synth-8gpu-72g`
  - Layout: 8 devices, 18 shards/device, 4 GiB/shard, 576 GiB total.
  - VAST generation with host zero writes + `fsync`: 576 GiB in 75.416s,
    7.64 GiB/s.
  - VAST generation repeat with host zero writes + `fsync`: 576 GiB in 69.495s,
    8.29 GiB/s.
  - Sharded SSD generation with host zero writes + `fsync`: 576 GiB in 37.985s,
    15.16 GiB/s.
- Clean up generated synthetic input data after final measurements if space is
  needed:
  - `/checkpoints/gms-synth-8gpu-72g`
  - `/mnt/nvme*/gms-synth-8gpu-72g`

## Open Items

- Decide whether to keep the benchmark-only NIXL POSIX sharded reader code or
  strip it before pushing the PR.
- Decide whether to make Sharded SSD use NIXL POSIX by default now that direct
  file descriptors make it performance-competitive with `preadv/O_DIRECT`.

## Local Validation

- `python3 -m py_compile lib/gpu_memory_service/snapshot/backends/pinned_staging.py lib/gpu_memory_service/snapshot/backends/nixl.py`
- `black` on the touched GMS snapshot backend/client/CLI files.
- `pre-commit run --files` on the touched GMS snapshot backend/client/CLI files,
  including `pinned_staging.py`.

## 2026-05-17 NIXL Substrate Refactor Validation

Refactor implemented:

- Added `nixl_common.py` for NIXL API loading, agent/backend creation, direct
  file descriptor opening, transfer polling, and resource release.
- Added `nixl_staging.py` for the shared NIXL POSIX FILE -> pinned DRAM -> VRAM
  staging path.
- Added `nixl_gds.py` for the NIXL GDS FILE -> VRAM path.
- Reduced `nixl.py` to the default PVC/VAST storage grouping policy.
- Reduced `sharded_ssd.py` to local SSD root parsing and grouping; transfer now
  uses the same NIXL POSIX staging path as the default backend.
- Removed the temporary `sharded_ssd_reader` config/CLI surface.

Local validation after refactor:

- `python3 -m py_compile` on the refactored backend/client/CLI files.
- `black` on the refactored backend/client/CLI files.
- `pre-commit run --files` on the refactored backend/client/CLI files.

nscale validation setup:

- Recreated `schwinns/gms-transfer-bench-s2877` on
  `cluster-0967a26d-pool-14bee067-prctr-s2877`.
- Copied the refactored source overlay to `/tmp/dynamo-src/lib`.
- Verified both VAST and local SSD test files opened with `O_DIRECT`.
- Temporarily scaled the same hwoo/QA graph worker components to zero for the
  8-GPU run.
- Restored those graph worker components to replicas=1, verified all three
  DynamoGraphDeployments reached Ready, and deleted the benchmark pod after the
  validation run.

Smoke checks:

| Backend | GPUs | GiB/GPU | Workers/GPU | Aggregate GiB/s | Notes |
|---|---:|---:|---:|---:|---|
| PVC/VAST NIXL POSIX | 1 | 4 | 1 | 8.09 | one VAST shard |
| Sharded SSD NIXL POSIX | 1 | 4 | 1 | 4.22 | one local shard/root; not representative |

Production-sized 8x72 GiB load after refactor:

| Backend | GPUs | Total GiB | Workers/GPU | Critical Path s | Aggregate GiB/s | Previous comparable result |
|---|---:|---:|---:|---:|---:|---:|
| Sharded SSD NIXL POSIX | 8 | 576 | 8 | 12.369 | 46.57 | 46.66 GiB/s |
| PVC/VAST NIXL POSIX | 8 | 576 | 8 | 25.569 | 22.53 | 21.15 GiB/s |

Interpretation:

- The NIXL substrate refactor preserved Sharded SSD restore performance. The
  post-refactor 46.57 GiB/s result matches the prior direct-NIXL Sharded SSD run
  at 46.66 GiB/s and remains in the same band as the earlier `preadv/O_DIRECT`
  plateau.
- PVC/VAST default NIXL also stayed in the same direct-I/O performance band,
  measuring 22.53 GiB/s versus the previous ~21-22 GiB/s plateau.
- The performance gap and recommendation are unchanged: Sharded SSD remains the
  fastest restore path for the 8-GPU 72 GiB/GPU target, while PVC/VAST NIXL
  remains the simpler default path when data lives on PVC.

## 2026-05-18 Handoff Snapshot

Use this section as the handoff index for the NIXL common-executor refactor and
the no-regression checks.

Workspace state:

- Worktree: `/private/tmp/dynamo-gms-transfer-mainbase`
- Git state: detached `HEAD`
- `HEAD`: `232b2e0f22a625b97419d7f783ee5e6d43d78bf7`
- Corresponding PR: `ai-dynamo/dynamo#9635`
  (`https://github.com/ai-dynamo/dynamo/pull/9635`)
- PR branch: `schwinns/gms-transfer-backends-draft`
- PR base: `main`
- Main repo checkout remains separate at `/Users/schwinns/dynamo` on `main`.
- The transfer worktree is dirty.

Dirty tracked files:

- `lib/gpu_memory_service/cli/storage_runner.py`
- `lib/gpu_memory_service/snapshot/backends/nixl.py`
- `lib/gpu_memory_service/snapshot/backends/sharded_ssd.py`
- `lib/gpu_memory_service/snapshot/transfer.py`

Untracked files:

- `gms-multigpu-benchmark-worklog.md`
- `lib/gpu_memory_service/snapshot/backends/nixl_common.py`
- `lib/gpu_memory_service/snapshot/backends/nixl_gds.py`
- `lib/gpu_memory_service/snapshot/backends/nixl_staging.py`

Latest code-shape summary:

- `nixl_common.py` owns NIXL import/agent helpers, direct read fd opening,
  generic transfer start/wait/release helpers, and the bounded in-flight
  transfer window for handle-level routes.
- `nixl_staging.py` owns the shared NIXL POSIX FILE -> pinned DRAM -> VRAM
  staging path used by both default PVC/VAST NIXL and Sharded SSD.
- `nixl_gds.py` owns the NIXL GDS_MT FILE -> VRAM route and now respects
  `GMSSnapshotConfig.max_workers` as a max in-flight transfer window.
- `nixl.py` is reduced to the default PVC/VAST grouping policy.
- `sharded_ssd.py` is reduced to local SSD root parsing and grouping, then
  reuses the shared NIXL POSIX staging path.

Local validation:

- `python3 -m py_compile` passed for the refactored GMS snapshot backend,
  client, and CLI files.
- `black` passed for the same touched file set.
- `pre-commit run --files` passed for the same touched file set.

Cluster validation:

- Kubernetes context:
  `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster`
- Node: `cluster-0967a26d-pool-14bee067-prctr-s2877`
- Benchmark pod used: `schwinns/gms-transfer-bench-s2877`
- Runtime image:
  `nvcr.io/nvidian/dynamo-dev/schwinns:dynamo-vllm-placeholder-quiesce-sm80sm100-tp2-dynsnapshot-filepg-rt4-ncclsm80-nosplit-nsrestore-20260514T204156Z`
- Source overlay copied into the pod:
  `/tmp/dynamo-src/lib/gpu_memory_service`
- Temporary local helper files used from outside the repo:
  `/tmp/gms-transfer-bench-s2877.yaml` and `/tmp/gms_multi_gpu_bench.py`
- Benchmark pod was deleted after the run.
- Restored worker components after the run and verified Ready with replicas=1:
  - `hwoo-snapshot/mx-gms-cold-qwen3-06b` component `VllmDecodeWorker`
  - `hwoo-snapshot/mx-gms-snapshot-qwen3-06b` component `VllmDecodeWorker`
  - `dynamo-qa-ci/sglang-agg-gms-rc1` component `decode`

No-regression result with the latest patch:

| Backend | GPUs | GiB/GPU | Workers/GPU | Critical Path s | Aggregate GiB/s | Prior comparable band |
|---|---:|---:|---:|---:|---:|---:|
| Sharded SSD NIXL POSIX `O_DIRECT` | 8 | 72 | 8 | 12.352 | 46.63 | 46.66-47.33 |
| PVC/VAST NIXL POSIX `O_DIRECT` | 8 | 72 | 8 | 25.888 | 22.25 | ~21-22 |

Interpretation for handoff:

- No performance regression was observed in either production-relevant restore
  path.
- The Sharded SSD result stayed in the previous plateau. The small difference
  versus 46.66-47.33 GiB/s is normal run-to-run noise.
- The PVC/VAST default NIXL result stayed in the previous direct-I/O band.
- The POSIX staging chunk loop was not moved behind a generic per-chunk callback,
  so the common executor refactor does not add Python dispatch overhead to the
  hot path.
- The shared bounded transfer helper is intended for handle-level routes,
  especially `nixl-gds` now and UCX P2P soon.

GDS caveat:

- A one-GPU `nixl-gds` smoke on local SSD failed before transfer start during
  NIXL GDS FILE registration:
  `GDS_MT: failed to create file handle: GDS_MT: file register error: error=5027`
  followed by `nixlBackendError: NIXL_ERR_BACKEND`.
- This did not produce a GDS throughput number and did not exercise the new
  bounded-window loop. Treat it as an environment/filesystem GDS support issue
  in this benchmark pod, not as a measured regression.

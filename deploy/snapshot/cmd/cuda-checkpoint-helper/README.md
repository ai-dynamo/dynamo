<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA/NIXL POSIX transfer benchmark

`cuda-nixl-posix-benchmark` is installed in the snapshot-agent image. It uses
the same pinned-slot transfer engine as `cuda-checkpoint-helper`, but owns its
CUDA allocation and stream so it can run independently of cuCheckpoint.

The benchmark supports checkpoint/write and restore/read. Pattern setup,
device clearing, and full byte verification happen outside each reported
transfer interval. Each measured run emits one JSON object on stdout; concise
progress and summary text goes to stderr.

```bash
cuda-nixl-posix-benchmark --help

cuda-nixl-posix-benchmark \
  --operation restore \
  --file /checkpoints/bench/cuda-posix.bin \
  --bytes $((8 * 1024 * 1024 * 1024)) \
  --device 0 \
  --transfer-buffer-count 2 \
  --transfer-chunk-bytes $((32 * 1024 * 1024)) \
  --file-count 1 \
  --warmups 1 \
  --iterations 5 \
  --cache-profile client-evicted \
  >restore.jsonl
```

For a layout sweep, use `--file-count 1`, `2`, `4`, or `8`. Multiple files are
deterministic contiguous shards of the logical extent, and JSON includes
per-file service metrics plus aggregate effective throughput.

```bash
for slots in 1 2 4 8; do
  for chunk_mib in 16 32 64 128; do
    cuda-nixl-posix-benchmark \
      --operation restore \
      --file /checkpoints/bench/cuda-posix.bin \
      --bytes $((8 * 1024 * 1024 * 1024)) \
      --transfer-buffer-count "${slots}" \
      --transfer-chunk-bytes $((chunk_mib * 1024 * 1024)) \
      --file-count 1 \
      --warmups 1 \
      --iterations 5 \
      --cache-profile buffered-warm \
      >"restore-s${slots}-c${chunk_mib}m.jsonl"
  done
done
```

> **Important:** `--file-count` does not add storage concurrency. This first
> pass has one owner, one NIXL agent, one synchronous storage request in flight,
> and one CUDA submitter. Multiple files alone might not improve bandwidth.
> Future storage lanes can feed the same bounded pinned ring without changing
> logical chunk scheduling or adding CUDA submitters.

Production defaults remain one 64 MiB slot per CUDA device. Pinned memory is
approximately:

```text
active devices * transfer buffer count * transfer chunk bytes
```

The helper and benchmark reject more than 1 GiB per device or 2 GiB per
operation. Use the smallest configuration on the throughput plateau.

## CustomStorage helper phase telemetry

Successful POSIX CustomStorage checkpoint and restore operations emit the
`cuda_custom_storage_transfer` JSON payload. `duration_seconds` and the
existing transfer service fields are unchanged for comparison with earlier
runs. The additional monotonic wall-clock fields are:

- `helper_main_to_telemetry_seconds`: helper `main` entry through the instant
  immediately before the JSON write.
- `custom_storage_total_seconds`: `DoCustomStorage` entry through that same
  instant.
- `storage_directory_validation_seconds`: storage directory preparation or
  validation. CustomStorage capability is resolved once at daemon startup via
  `cuGetProcAddress` for CUDA API 13.4.
- `cuda_init_seconds`, `cuda_device_count`,
  `device_enumeration_seconds`, and `primary_context_retain_seconds`: CUDA
  initialization, count discovery, and aggregate device lookup/context retain.
- `manifest_validation_seconds` and `device_map_preparation_seconds`:
  restore manifest read/extent-file validation and restore mapping parse.
  Their checkpoint values measure the corresponding no-op preparation.
- `cuda_process_api_seconds`: the exact synchronous
  `cuCheckpointProcessCheckpoint` or `cuCheckpointProcessRestore` call.
- `metadata_job_construction_seconds`: returned CustomStorage metadata checks,
  context/device/UUID lookup, extent mapping, pinned-memory calculation, and
  transfer-job construction.
- `worker_orchestration_seconds`: one wall interval from worker setup through
  all worker joins.
- `post_transfer_validation_seconds`: checkpoint extent validation and
  manifest publication; restore reports its no-op interval.
- `cuda_operation_complete_seconds`: the exact
  `cuCheckpointOperationComplete` call.
- `primary_context_release_seconds`: explicit release of all retained primary
  contexts after successful CUDA acknowledgment and before telemetry.
- `primary_context_release_success` and `primary_context_release_status`:
  whether that one explicit release pass succeeded and its CUDA status code.
  A release failure is warned and reported after the operation has already
  been acknowledged; it is not retried.

Totals contain subphases and must not be added to them. Worker service fields
are sums across workers and can overlap each other in wall time. The
`timing_scope` JSON field records these rules. On the Go side, daemon RPC wall
duration is compared with `helper_main_to_telemetry_duration`. Derived fields
are emitted only when the expected timing value is present, finite,
nonnegative, and consistent with the RPC wall duration. Otherwise, the log
retains the raw wall duration and reports `helper_telemetry_status` plus
`helper_telemetry_error`.

Restore runs require an explicit cache interpretation:

- `--cache-profile buffered-warm` is the default. Restore files are prepared
  with buffered writes and `fsync`; warmups and measured runs then intentionally
  retain the client page-cache state. With `--warmups 1` or greater, the last
  warmup intentionally establishes the measured buffered-warm state. The
  exception is `--existing-file` with no explicit `--warmups`: its effective
  warmup count is zero, so measured iteration 0 does not claim buffered-warm
  residency. After iteration 0 succeeds and reads the complete file, later
  measured iterations report the resulting buffered-warm state.
- `--cache-profile client-evicted` calls
  `posix_fadvise(..., POSIX_FADV_DONTNEED)` for every prepared file after setup
  and again before every restore transfer. This reapplies eviction after each
  warmup, so measured iterations request client eviction even when warmups are
  enabled.

Client eviction is best effort, not a true cold-storage claim. The kernel may
retain pages despite accepting the advice, and VAST or another storage system's
server-side cache may remain warm. The benchmark never uses global
`drop_caches`. JSON records `cache_profile`, whether client eviction was
requested, how many per-file advisory calls the kernel accepted, and that
client/server residency is unknown where appropriate. Preserve these fields
when comparing runs.

Restore setup validates file existence, regular-file type, exact size, and
layout without rereading all file bytes before timing. Full deterministic
corruption detection is retained: restored GPU data is checked byte-for-byte
after every untimed warmup or measured transfer. Checkpoint results use
`cache_profile: "not-applicable"` and continue to verify the written files
outside the timed interval. Passing `--cache-profile` explicitly for a
checkpoint/write is rejected.

## Cross-node existing-file restore

Use `--existing-file` to restore bytes written by a separate benchmark
invocation. This option is restore-only. It opens every configured shard
read-only with symlink following disabled and validates regular-file type and
exact size. It never creates, truncates, or rewrites storage. JSON records
`storage_prepared_by_invocation`, `existing_file`, the effective `warmups`,
whether warmups were explicit, and
`client_cache_preconditioned_by_invocation` for both successful and failed
measured iterations. All restore transfers, including the normal prepared-file
mode and production helper restores, use `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`;
checkpoint transfers remain writable and create/truncate their destination.
The pinned NIXL POSIX backend accepts the supplied read-only descriptor for
fd-mode registration and performs `NIXL_READ` as file reads.

The benchmark pattern is defined over logical byte offsets. For each
eight-byte word, apply the SplitMix64 finalizer to
`word_index + 0x9e3779b97f4a7c15`, and store the resulting `uint64_t` in
little-endian byte order. Sharding does not restart the pattern: shard ranges
retain their logical offsets. Checkpoint/write initializes all GPU bytes with
this pattern; restore/read verifies every restored GPU byte against it.

The following sequence preserves the files across nodes without a reader-side
rewrite. Use the same `--file`, `--bytes`, and `--file-count` values on both
nodes.

On node A, write and verify the deterministic storage file:

```bash
cuda-nixl-posix-benchmark \
  --operation write \
  --file /checkpoints/bench/cross-node.bin \
  --bytes $((8 * 1024 * 1024 * 1024)) \
  --file-count 1 \
  --warmups 0 \
  --iterations 1 \
  >node-a-write.jsonl
```

After that command exits, unmount/detach the RWO PVC from node A using the
normal cluster storage workflow. Attach and mount the same PVC on previously
unused node B. Then read the existing bytes without preparation:

```bash
cuda-nixl-posix-benchmark \
  --operation read \
  --file /checkpoints/bench/cross-node.bin \
  --bytes $((8 * 1024 * 1024 * 1024)) \
  --file-count 1 \
  --existing-file \
  --warmups 0 \
  --iterations 1 \
  --cache-profile client-evicted \
  >node-b-read.jsonl
```

Omitting `--warmups` in `--existing-file` mode defaults to zero and avoids an
intentional pre-measurement read on node B. The explicit `--warmups 0` above
makes that contract visible in the captured command. Explicit values greater
than zero are allowed and perform verified untimed restores; with
`buffered-warm`, that establishes the reported client-cache state.
With existing-file, zero warmups, and `buffered-warm`, JSON reports
`client_cache_residency: "unknown-not-preconditioned"` for measured iteration
0 rather than claiming a warm cache. Once that iteration succeeds and reads
the complete file, later measured iterations report `buffered-warm`.
`client-evicted` is still best-effort; prove wire traffic by collecting NFS
mount/RPC, TCP port 2049, and relevant NIC or bond-member counter deltas around
the node B command.

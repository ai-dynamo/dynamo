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

Restore runs require an explicit cache interpretation:

- `--cache-profile buffered-warm` is the default. Restore files are prepared
  with buffered writes and `fsync`; warmups and measured runs then intentionally
  retain the client page-cache state. With `--warmups 1` or greater, the last
  warmup intentionally establishes the measured buffered-warm state.
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
outside the timed interval.

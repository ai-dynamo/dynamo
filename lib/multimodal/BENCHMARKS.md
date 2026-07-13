<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video preprocessing microbenchmarks

This benchmark intentionally starts with already-decoded, contiguous RGB8
frames. It excludes fetch, video decode, transport, and backend adaptation so
that changes measure only model preprocessing.

Run the maintained harness with:

```bash
cargo build --release -p dynamo-multimodal --example bench_qwen3_vl_video
RAYON_NUM_THREADS=8 \
  target/release/examples/bench_qwen3_vl_video 2000 512 512 8
```

The positional arguments are iterations, width, height, and frame count. The
harness reports both the model-family core and the complete `ProcessedMedia`
contract path, together with output shape and bytes to prevent comparing
different visual-token workloads.

## 2026-07-08 baseline and optimization log

Host: Intel Xeon Platinum 8480CL. The representative E2E workload is eight
512x512 frames and produces FP32 `[4096, 1536]` (25,165,824 bytes). Each number
below is a warm median from a release build.

| Change | Threads | Median | Decision |
|---|---:|---:|---|
| Zero-initialized output baseline | 1 | 3.98 ms | Replaced |
| Zero-initialized output baseline | 8 | 2.10 ms | Replaced |
| Write-once `MaybeUninit` output | 1 | 3.04 ms | Keep |
| Write-once `MaybeUninit` output | 8 | 0.41 ms | Keep |
| Explicit indexed pixel loop | 8 | 0.56 ms | Reverted: 38% slower |
| Flattened patch dispatch | 8 | 0.45-0.47 ms | Reverted: 8-12% slower |

The accepted output-allocation change removes a redundant 25 MiB serial
zero-fill. It also lets the parallel writers first-touch their own pages.
Release tests include a bit-exact fingerprint generated with Transformers
4.57.1, so the optimization cannot silently change Hugging Face output.

After that change, `perf` attributes about 88% of the 512x512 workload to the
actual RGB-to-Qwen patch layout. The complete contract wrapper is within
measurement noise of the raw processor core (less than 0.02 ms).

## Resize workload

Eight 1280x720 frames are aligned to Qwen's spatial factor and produce FP32
`[14080, 1536]` (86,507,520 bytes). At eight threads the median is about
15.5-16.5 ms. `perf` attributes 44% of cycles to the Pillow-compatible vertical
bicubic pass and 14% to patchification.

Increasing the vertical resampler accumulator width from four pixels to eight
did not improve the result and was reverted. Any future resize optimization
must retain the existing Hugging Face bit-exact fingerprint and should add
resized high-resolution fixtures before it is accepted.

## 2026-07-09 follow-up candidates

Runs were pinned to physical cores 56-63 and candidates were interleaved with
the saved baseline binary. A manual AVX2 gather for RGB-to-normalized-FP32
patchification improved the eight-thread 512x512 core median from about 0.425
ms to 0.387 ms, but regressed the one-thread median from 2.265 ms to 2.609 ms
(15%). It was reverted because concurrency-one latency is a required target.

Skipping the zero fill of Pillow resize scratch buffers reduced the one-thread
1280x720 median by roughly 0.5-0.7 ms, but produced no repeatable eight-thread
improvement and required an additional unsafe initialization invariant. It was
also reverted.

The resize workload scales from 77.4 ms at one thread to 47.2, 26.0, 16.4,
and 11.1 ms at 2, 4, 8, and 16 threads respectively. This confirms that the
current row and temporal-group parallelism is effective. The next worthwhile
code target is the Pillow-exact vertical convolution itself; allocation and
coefficient setup are below the threshold for added complexity.

The vertical-convolution follow-up tested two additional changes. Narrowing
fixed-point coefficients and accumulators from 64 to 32 bits did not improve
the eight-thread result and was reverted. Re-expressing each four-pixel source
block as one bounded RGB slice retained the exact accumulation order while
removing repeated address arithmetic and bounds checks. It reduced the
one-thread 1280x720 core median from about 73.28 ms to 70.63 ms (3.6%). A
longer eight-thread run reduced the run-level core median from 17.51 ms to
17.24 ms (1.5%); complete-path results at eight threads remained within
measurement noise. The safe slice change was retained because it improves the
concurrency-one target without an architecture-specific path or unsafe code.

Raw command outputs and `perf.data` files from this run live under
`target/video-preprocess-microbench/`; they are local artifacts and are not
source-controlled.

## 2026-07-09 bounded output slice and raw Hugging Face comparison

This continuation was pinned to CPU 84 for one thread and physical CPUs 84-91
for eight threads. Each Rust candidate was interleaved with the saved release
baseline. Re-expressing each four-pixel output block as one bounded mutable RGB
slice allows the compiler to eliminate repeated output bounds checks. It keeps
the same accumulation and clipping order, uses no unsafe code, and was retained.

| Path | Threads | Core median | Complete contract median | Delta |
|---|---:|---:|---:|---:|
| Saved safe-slice baseline | 1 | 76.542 ms | 76.468 ms | - |
| Bounded input and output slices | 1 | 66.482 ms | 66.438 ms | -13.1% |
| Saved safe-slice baseline | 8 | 16.326 ms | 17.078 ms | - |
| Bounded input and output slices | 8 | 15.085 ms | 15.255 ms | -10.7% complete path |

Four other vertical-resize candidates were measured at one thread and reverted:

| Candidate | Baseline | Candidate | Decision |
|---|---:|---:|---|
| Reusable resize coefficients/plan | 76.512 ms | 88.388 ms | Reverted: 15.5% slower |
| Specialized 4/5/6-tap loops | 77.072 ms | 93.500 ms | Reverted: 21.3% slower |
| Return raw resize storage directly | 76.493 ms | 77.914 ms | Reverted: 1.9% slower |
| Unsafe source-pointer loads | 76.550 ms | 84.134 ms | Reverted: 9.9% slower |

The raw Hugging Face comparison uses the same deterministic decoded RGB8
frames, processor configuration, output shape `[14080, 1536]`, output size
86,507,520 bytes, and grid `[4, 44, 80]`. Fetch and decode are excluded. The HF
measurement is the complete `Qwen3VLVideoProcessor(..., return_tensors="pt")`
call from Transformers 4.57.1 with Pillow 12.2.0. The Rust measurement includes
the complete backend-neutral `ProcessedMedia` contract path.

| Implementation | Threads | Median | Rust speedup |
|---|---:|---:|---:|
| Raw Hugging Face | 1 | 224.106 ms | - |
| Dynamo Rust complete path | 1 | 66.438 ms | 3.37x |
| Raw Hugging Face | 8 | 85.833 ms | - |
| Dynamo Rust complete path | 8 | 15.255 ms | 5.63x |

The maintained HF harness is
`scripts/benchmark_qwen3_vl_video_hf.py`. It keeps input construction outside
the timed region and reports dependency versions, CPU affinity, tensor shape,
bytes, and grid so incompatible workloads are not compared. Raw outputs are in
`target/video-preprocess-microbench/hf-4.57.1-1280x720x8-{1t,8t}.txt`.

### Default-normalization bit parity

Whole-output fingerprinting exposed a one-ULP mismatch in the original
normalization LUT. It had algebraically folded normalization into
`pixel * scale + bias`, while the HF fast processor first fuses 1/255 into its
FP32 mean and standard deviation and then evaluates `(pixel - mean) / std`.
The LUT now follows that operation order; patchification still performs one
lookup per value.

The normalized FP32 FNV-1a fingerprints now match for both representative
workloads:

| Workload | Output bytes | HF 4.57.1 | Dynamo Rust |
|---|---:|---|---|
| 512x512x8, no resize | 25,165,824 | `c2b133ed08644325` | `c2b133ed08644325` |
| 1280x720x8, resize | 86,507,520 | `834a85cb7ac57125` | `834a85cb7ac57125` |

The version-pinned unit fixture now exercises default normalization instead of
disabling it. The generator also recovers and fingerprints the source pixels
from normalized values, retaining an independent resize/layout check.

Interleaved pre/post release binaries showed no performance regression:

| Workload | Threads | Before complete path | After complete path |
|---|---:|---:|---:|
| 512x512x8 | 1 | 2.376 ms | 2.360 ms |
| 1280x720x8 | 1 | 66.440 ms | 65.332 ms |
| 1280x720x8 | 8 | 15.808 ms | 15.576 ms |

Pass `--fingerprint` to the HF harness and set
`DYNAMO_BENCH_FINGERPRINT=1` for the Rust harness to hash the complete FP32
output outside the timed region.

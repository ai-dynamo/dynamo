<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend Video Decode Experiments

This directory records experimental frontend video decode work on the
`jegu/opencv-video-decoder` branch. It is intended for sharing results and
reproducing decisions; this branch is not planned as a pull request.

## Run

The benchmark generates a deterministic 1280x720, 24 FPS, 10-second H.264
fixture, decodes 30 uniformly sampled RGB frames, and measures batches at
concurrency 1, 8, and 32. Each case has one warmup and seven measured runs;
the reported value is the median batch elapsed time.

```bash
OPENCV_PREFIX=/opt/opencv-4.13.0 \
    benchmarks/multimodal/video_decode/run.sh
```

The first release/LTO build can take about ten minutes. Raw logs, fixture
metadata, and the rendered summary default to
`/tmp/dynamo-video-decode-benchmark`.

Useful overrides:

```bash
CPUSET=0-7 \
ITERATIONS=5 \
CONCURRENCIES=1,8 \
NUM_FRAMES=30 \
VIDEO=/path/to/video.mp4 \
RESULTS_DIR=/tmp/video-results \
    benchmarks/multimodal/video_decode/run.sh
```

Set `REGENERATE_FIXTURE=1` to overwrite the generated fixture. `ITERATIONS`
must be odd so that the median is unambiguous.

## Reference Environment

- Date: 2026-07-06
- CPU: Arm Neoverse-V2, pinned to CPUs 0-31
- OpenCV: 4.13.0, built with FFmpeg and without GStreamer
- FFmpeg libraries: 6.1
- Rust: 1.93.1
- Input: H.264, 1280x720, yuv420p, 24 FPS, 10 seconds, 240 frames
- Output: RGB `u8`, shape `[30, 720, 1280, 3]`

## Results

Median batch latency from seven measured runs:

| Backend | C1 | C8 | C32 |
| --- | ---: | ---: | ---: |
| Original `video_rs` | 683.0 ms | 692.6 ms | 772.3 ms |
| Optimized `ffmpeg` | 72.3 ms | 183.5 ms | 548.5 ms |
| Optimized `opencv` | 70.2 ms | 286.9 ms | 763.6 ms |

Throughput from the same runs:

| Backend | C1 | C8 | C32 |
| --- | ---: | ---: | ---: |
| Original `video_rs` | 1.464 video/s | 11.551 video/s | 41.437 video/s |
| Optimized `ffmpeg` | 13.824 video/s | 43.603 video/s | 58.341 video/s |
| Optimized `opencv` | 14.252 video/s | 27.887 video/s | 41.904 video/s |

Compared with `video_rs`, optimized FFmpeg reduced batch latency by 89.4%,
73.5%, and 29.0% at C1, C8, and C32. Optimized OpenCV reduced it by 89.7%,
58.6%, and 1.1% respectively.

These results describe one host and one CFR H.264 workload. They are not a
general codec or platform comparison.

## FFmpeg Experiments

### Convert only sampled frames to RGB

`video-rs` creates an RGB scaler as part of its decoder and converts every
decoded frame to RGB. This workload must decode about 240 compressed frames
but retains only 30, so most RGB conversions are discarded.

The experimental `ffmpeg` backend decodes all frames in their native YUV
format, preserving inter-frame codec dependencies, and invokes swscale only
for selected frames. This optimization alone reduced latency by about 30% at
all three concurrency levels. It is Dynamo-specific and was inspired by, but
is not an implementation from, SMG PR #1865.

### Decoder threads

Fixed thread sweeps found different optima by concurrency:

| Concurrency | Best FFmpeg decoder threads |
| --- | ---: |
| C1 | 16 |
| C8 | 8 |
| C32 | 1 |

The adaptive experiment counts active decodes, waits 1 ms to coalesce request
bursts, and assigns a bounded thread count. This produced the final FFmpeg
numbers above. The policy needs broader testing under CPU quotas and with more
codecs before being treated as generally optimal.

### Direct RGB output

Writing swscale output directly into the final tensor buffer was neutral at
C1 and improved C32 by about 1.3%. The gain is small relative to the added
unsafe borrowed-frame handling.

## OpenCV Experiments

### Sequential `grab()` and sampled `read()`

The original implementation set `CAP_PROP_POS_FRAMES` before every sample.
For inter-frame codecs such as H.264, each seek flushes and repositions the
decoder. The optimized path calls `grab()` for intervening frames and `read()`
only for sampled frames.

This was the largest OpenCV improvement:

| Concurrency | Repeated seek | Sequential grab/read | Reduction |
| --- | ---: | ---: | ---: |
| C1 | 436.7 ms | 71.7 ms | 83.6% |
| C8 | 1235.9 ms | 283.6 ms | 77.1% |
| C32 | 4250.1 ms | 822.5 ms | 80.6% |

Sparse sampling still benefits from seeking. On this fixture, sequential
advance won with about 59 intervening frames, while seeking won with about 79.
The implementation therefore falls back to seek when the gap exceeds 64.

Duplicate sample indices copy the preceding output frame instead of decoding
the same source frame again.

### Decoder threads

The OpenCV sweep had a different boundary from direct FFmpeg:

| Concurrency | Best OpenCV decoder threads |
| --- | ---: |
| C1 | 16 |
| C8 | 8-16 |
| C32 | 2 |

The OpenCV adaptive policy mainly helped C32, by about 6%, and was neutral at
C1 and C8. This backend-specific result is why one shared thread formula
should not be assumed to work for every video backend.

### In-memory input

OpenCV input was changed from a named temporary file to the same Linux memfd
pattern already used by the FFmpeg path. Performance was within normal run
variance (about +/-2%), but memfd avoids temporary-file lifecycle and disk
filesystem dependence.

### Direct RGB output

Converting BGR directly into the final output slice made C1 about 14.8% slower
and improved C8/C32 by only 4.3%/1.5%. It remains an experiment and is disabled
by default.

Output tensor allocation was already preallocated before these experiments,
so preallocation is not counted as a new optimization.

## Correctness

For the reference fixture, `video_rs`, optimized FFmpeg, baseline OpenCV, and
optimized OpenCV produced byte-identical output:

```text
82,944,000 bytes compared
differing bytes: 0
maximum absolute difference: 0
mean absolute difference: 0.0
```

A smaller checked-in 240p fixture also has a non-ignored pixel-parity test for
the `video_rs` and optimized FFmpeg paths.

Coverage still needed before production use includes VFR and missing PTS,
H.265/VP9/AV1, 10-bit and YUV422/YUV444 inputs, full versus limited color
range, BT.601/BT.709/BT.2020, rotation metadata, odd dimensions, corrupted
streams, and CPU/cgroup variations.

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

VIDEO="${VIDEO:-/tmp/dynamo-video-720p24-10s.mp4}"
CPUSET="${CPUSET:-0-31}"
ITERATIONS="${ITERATIONS:-7}"
CONCURRENCIES="${CONCURRENCIES:-1,8,32}"
NUM_FRAMES="${NUM_FRAMES:-30}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/dynamo-video-decode-benchmark}"
OPENCV_PREFIX="${OPENCV_PREFIX:-/opt/opencv-4.13.0}"

for command in cargo ffmpeg ffprobe taskset; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "required command not found: ${command}" >&2
        exit 1
    fi
done

if (( ITERATIONS < 1 || ITERATIONS % 2 == 0 )); then
    echo "ITERATIONS must be a positive odd number" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${VIDEO}")" "${RESULTS_DIR}"

if [[ ! -s "${VIDEO}" || "${REGENERATE_FIXTURE:-0}" == "1" ]]; then
    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -f lavfi \
        -i "testsrc2=size=1280x720:rate=24:duration=10" \
        -c:v libx264 \
        -preset veryfast \
        -crf 23 \
        -pix_fmt yuv420p \
        -g 48 \
        -an \
        "${VIDEO}"
fi

ffprobe \
    -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames \
    -show_entries format=duration \
    -of default=noprint_wrappers=1 \
    "${VIDEO}" | tee "${RESULTS_DIR}/fixture.txt"

if [[ -d "${OPENCV_PREFIX}" ]]; then
    export PKG_CONFIG_PATH="${OPENCV_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${OPENCV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${REPO_ROOT}"

cargo test \
    -p dynamo-llm \
    --release \
    --features media-opencv-video \
    --lib \
    bench_video_decode \
    --no-run

run_backend() {
    local backend="$1"
    local log_file="${RESULTS_DIR}/${backend}.log"

    echo "Running ${backend}..."
    taskset -c "${CPUSET}" env \
        -u DYN_BENCH_VIDEO_THREADS \
        -u DYN_BENCH_VIDEO_COALESCE_MS \
        -u DYN_BENCH_FFMPEG_ADAPTIVE_THREADS \
        -u DYN_BENCH_FFMPEG_DIRECT_OUTPUT \
        -u DYN_BENCH_OPENCV_ADAPTIVE_THREADS \
        -u DYN_BENCH_OPENCV_MEMFILE \
        -u DYN_BENCH_OPENCV_SEQUENTIAL \
        -u DYN_BENCH_OPENCV_GRAB_LIMIT \
        -u DYN_BENCH_OPENCV_DIRECT_OUTPUT \
        DYN_BENCH_VIDEO="${VIDEO}" \
        DYN_BENCH_VIDEO_BACKEND="${backend}" \
        DYN_BENCH_ITERATIONS="${ITERATIONS}" \
        DYN_BENCH_CONCURRENCIES="${CONCURRENCIES}" \
        DYN_BENCH_NUM_FRAMES="${NUM_FRAMES}" \
        cargo test \
            -p dynamo-llm \
            --release \
            --features media-opencv-video \
            --lib \
            bench_video_decode \
            -- \
            --ignored \
            --nocapture \
            --test-threads=1 \
        2>&1 | tee "${log_file}" | awk '/VIDEO_BENCH|test result/'
}

run_backend video_rs
run_backend ffmpeg
run_backend opencv

awk '
/VIDEO_BENCH/ {
    backend = ""
    concurrency = ""
    median_ms = ""
    videos_per_second = ""
    for (i = 1; i <= NF; i++) {
        split($i, pair, "=")
        if (pair[1] == "backend") backend = pair[2]
        if (pair[1] == "concurrency") concurrency = pair[2]
        if (pair[1] == "median_ms") median_ms = pair[2]
        if (pair[1] == "videos_per_second") videos_per_second = pair[2]
    }
    latency[backend, concurrency] = median_ms
    throughput[backend, concurrency] = videos_per_second
}
END {
    names[1] = "VideoRs"
    names[2] = "Ffmpeg"
    names[3] = "OpenCv"
    labels[1] = "original video_rs"
    labels[2] = "optimized ffmpeg"
    labels[3] = "optimized opencv"
    concurrencies[1] = 1
    concurrencies[2] = 8
    concurrencies[3] = 32

    print ""
    print "Median batch latency"
    printf "%-20s %12s %12s %12s\n", "Backend", "C1", "C8", "C32"
    for (i = 1; i <= 3; i++) {
        printf "%-20s", labels[i]
        for (j = 1; j <= 3; j++) {
            printf " %9.1f ms", latency[names[i], concurrencies[j]]
        }
        print ""
    }

    print ""
    print "Throughput"
    printf "%-20s %12s %12s %12s\n", "Backend", "C1", "C8", "C32"
    for (i = 1; i <= 3; i++) {
        printf "%-20s", labels[i]
        for (j = 1; j <= 3; j++) {
            printf " %8.3f vid/s", throughput[names[i], concurrencies[j]]
        }
        print ""
    }
}
' "${RESULTS_DIR}/video_rs.log" \
  "${RESULTS_DIR}/ffmpeg.log" \
  "${RESULTS_DIR}/opencv.log" | tee "${RESULTS_DIR}/summary.txt"

echo "Raw results: ${RESULTS_DIR}"

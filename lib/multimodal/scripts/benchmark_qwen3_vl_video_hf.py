# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the decode-free Hugging Face Qwen3-VL video processor."""

import argparse
import os
import statistics
import time

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


def make_frames(width: int, height: int, frame_count: int) -> list[Image.Image]:
    """Build the same deterministic RGB8 input as the Rust microbenchmark."""
    indices = np.arange(width * height * 3, dtype=np.uint32)
    return [
        Image.fromarray(
            ((indices + frame * 17) & 0xFF).astype(np.uint8).reshape(height, width, 3)
        )
        for frame in range(frame_count)
    ]


def fnv1a(data: memoryview) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iterations", nargs="?", type=int, default=20)
    parser.add_argument("width", nargs="?", type=int, default=224)
    parser.add_argument("height", nargs="?", type=int)
    parser.add_argument("frame_count", nargs="?", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fingerprint", action="store_true")
    args = parser.parse_args()
    height = args.height if args.height is not None else args.width
    if min(args.iterations, args.width, height, args.frame_count, args.threads) <= 0:
        parser.error(
            "iterations, dimensions, frame count, and threads must be positive"
        )

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    frames = make_frames(args.width, height, args.frame_count)
    processor = Qwen3VLVideoProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        size={"shortest_edge": 65_536, "longest_edge": 16_777_216},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        resample=Image.Resampling.BICUBIC,
        do_sample_frames=False,
    )

    def preprocess() -> transformers.BatchFeature:
        return processor(
            videos=[frames],
            do_resize=True,
            do_normalize=True,
            do_sample_frames=False,
            return_tensors="pt",
        )

    output = preprocess()
    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        output = preprocess()
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    values = output["pixel_values_videos"]
    grid = output["video_grid_thw"][0].tolist()
    print(
        "hugging-face qwen3-vl video preprocess: "
        f"transformers={transformers.__version__}, pillow={PIL.__version__}, "
        f"frames={args.frame_count}, size={args.width}x{height}, threads={args.threads}, "
        f"affinity={len(os.sched_getaffinity(0))}, output_shape={list(values.shape)}, "
        f"output_bytes={values.numel() * values.element_size()}, grid={grid}, "
        f"iterations={args.iterations}, processor_median_ms={statistics.median(samples_ms):.3f}, "
        f"processor_mean_ms={statistics.fmean(samples_ms):.3f}"
    )
    if args.fingerprint:
        fp32_bytes = memoryview(values.numpy()).cast("B")
        print(f"fnv1a_fp32={fnv1a(fp32_bytes)}")


if __name__ == "__main__":
    main()

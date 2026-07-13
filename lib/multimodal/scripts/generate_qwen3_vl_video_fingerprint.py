# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the pinned Qwen3-VL video FP32 and pixel fingerprints."""

import numpy as np
import PIL
import transformers
from PIL import Image
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    smart_resize,
)


def make_frame(width: int, height: int, seed: int) -> Image.Image:
    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.stack(
        (
            (seed + x * 7 + y * 3) % 256,
            (seed + x * 5 + y * 11) % 256,
            (seed + x + y * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def fnv1a(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def main() -> None:
    print(f"transformers={transformers.__version__}")
    print(f"pillow={PIL.__version__}")
    width, height = 37, 35
    seeds = (3, 101, 177)
    target_height, target_width = smart_resize(
        len(seeds),
        height,
        width,
        temporal_factor=2,
        factor=32,
        min_pixels=65536,
        max_pixels=16777216,
    )
    frames = [
        make_frame(width, height, seed).resize(
            (target_width, target_height), Image.Resampling.BICUBIC
        )
        for seed in seeds
    ]
    processor = Qwen3VLVideoProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        resample=Image.Resampling.BICUBIC,
        do_sample_frames=False,
    )
    output = processor(
        videos=[frames],
        do_resize=False,
        do_normalize=True,
        do_sample_frames=False,
        return_tensors="pt",
    )
    values = np.ascontiguousarray(output["pixel_values_videos"].cpu().numpy())
    pixel_values = np.rint((values * 0.5 + 0.5) * 255.0).astype(np.uint8)
    print(f"shape={list(values.shape)}")
    print(f"grid={output['video_grid_thw'][0].tolist()}")
    print(f"fnv1a_fp32={fnv1a(values.tobytes())}")
    print(f"fnv1a_pixel_u8={fnv1a(pixel_values.tobytes())}")


if __name__ == "__main__":
    main()

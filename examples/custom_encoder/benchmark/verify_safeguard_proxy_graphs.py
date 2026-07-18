# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify benchmark Qwen CUDA graph parity and memory stability."""

from __future__ import annotations

import argparse
import logging
import os

import torch
from PIL import Image

from examples.custom_encoder.qwen2_5_vl_benchmark_encoder import (
    _GRAPH_IMAGE_SIZES,
    Qwen2_5VLBenchmarkEncoder,
    _StaticQwen2VLVisionForward,
)


def verify(
    model: str,
    encoder_model: str,
    output_hidden_size: int,
    replay_iterations: int,
) -> None:
    os.environ["DYN_QWEN2_VL_ENCODER_MODEL"] = encoder_model
    os.environ["DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE"] = str(output_hidden_size)
    encoder = Qwen2_5VLBenchmarkEncoder()
    encoder.build(model)
    try:
        buckets = encoder.buckets
        assert buckets is not None
        templates = []
        for index, size in enumerate(_GRAPH_IMAGE_SIZES):
            for variant in range(2):
                image = Image.new(
                    "RGB",
                    size,
                    color=(
                        ((index + variant) * 71) % 256,
                        ((index + 2 * variant) * 113) % 256,
                        127,
                    ),
                )
                templates.append(encoder._process_image(image))

        retained = None
        for item in templates:
            grid = encoder._grid_key(item)
            padding_item = type(item)(
                pixel_values=torch.zeros_like(item.pixel_values),
                image_grid_thw=item.image_grid_thw.clone(),
            )
            for bucket in buckets:
                padded_items = [item] + [padding_item] * (bucket - 1)
                adapter = _StaticQwen2VLVisionForward(
                    encoder._visual,
                    grid,
                    bucket,
                    encoder._device,
                    encoder._require_output_hidden_size(),
                ).eval()
                with torch.inference_mode():
                    adapter_output = adapter(
                        torch.cat(
                            [padded.pixel_values for padded in padded_items], dim=0
                        ).to(device=encoder._device, dtype=encoder._visual.dtype)
                    ).cpu()
                native_padded_output = torch.cat(
                    encoder._forward_eager(padded_items), dim=0
                )
                torch.testing.assert_close(
                    adapter_output, native_padded_output, rtol=1e-2, atol=1e-2
                )
                print(f"adapter_parity_ok grid={grid} bucket={bucket}")

                # Exercise padding at the bottom, middle, and top of every rung.
                # Checking every count through bucket 64 would repeat thousands
                # of full 32-block ViT forwards without increasing graph coverage.
                real_counts = sorted({1, max(1, bucket // 2), bucket})
                for real_count in real_counts:
                    items = [item] * real_count
                    eager = encoder._forward_eager(items)
                    # The graph always executes its static bucket shape. Compare
                    # against the native tower at that exact padded shape first;
                    # FlashAttention can select a different bf16 kernel for an
                    # unpadded call.
                    padded_eager = encoder._forward_eager(
                        items + [padding_item] * (bucket - real_count)
                    )[:real_count]
                    graphed = encoder._forward_graph(items, bucket)
                    assert len(eager) == len(graphed) == real_count
                    for expected, padded_expected, actual in zip(
                        eager, padded_eager, graphed
                    ):
                        torch.testing.assert_close(
                            actual, padded_expected, rtol=1e-2, atol=1e-2
                        )
                        # Bound the normal padded-vs-unpadded bf16 drift as a
                        # separate check without weakening graph replay parity.
                        torch.testing.assert_close(
                            actual, expected, rtol=5e-2, atol=5e-1
                        )
                    if retained is None:
                        # Retain the actual returned split views. The later replay
                        # loop must prove their shared CPU base is independent of
                        # the static graph output; cloning here would make that
                        # lifetime check a false positive.
                        retained = list(graphed)
                    print(
                        f"parity_ok grid={grid} real_count={real_count} bucket={bucket}"
                    )

        assert retained is not None
        retained_snapshot = [tensor.clone() for tensor in retained]
        torch.cuda.synchronize()
        reserved_before = torch.cuda.memory_reserved()
        for iteration in range(replay_iterations):
            item = templates[iteration % len(templates)]
            bucket = buckets[iteration % len(buckets)]
            real_count = iteration % bucket + 1
            encoder._forward_graph([item] * real_count, bucket)
        torch.cuda.synchronize()
        reserved_after = torch.cuda.memory_reserved()
        for expected, actual in zip(retained_snapshot, retained):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert reserved_after == reserved_before, (
            "CUDA allocator grew during steady graph replay: "
            f"before={reserved_before} after={reserved_after}"
        )
        pinned_staging_bytes = sum(
            entry.host_pixel_values.numel() * entry.host_pixel_values.element_size()
            for entry in encoder._graphs.values()
        )
        print(
            "memory_plateau_ok "
            f"iterations={replay_iterations} reserved_bytes={reserved_after} "
            f"pinned_staging_bytes={pinned_staging_bytes}"
        )
        if output_hidden_size != 2048:
            print(
                "parity_scope=performance-only "
                f"native_vision_width=2048 truncated_output_width={output_hidden_size} "
                "quality_or_model_parity_claim=false"
            )

    finally:
        encoder.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--encoder-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-hidden-size", type=int, default=2048)
    parser.add_argument("--replay-iterations", type=int, default=20)
    args = parser.parse_args()
    verify(
        args.model,
        args.encoder_model,
        args.output_hidden_size,
        args.replay_iterations,
    )


if __name__ == "__main__":
    main()

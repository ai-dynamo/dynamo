# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Qwen3-VL custom-encoder CUDA graph parity and memory stability."""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import replace

import torch
from PIL import Image

from examples.custom_encoder.qwen3_vl_vision_encoder import (
    _GRAPH_IMAGE_SIZES,
    Qwen3VLVisionEncoder,
    _StaticQwen3VLVisionForward,
)


def verify(model: str, replay_iterations: int) -> None:
    encoder = Qwen3VLVisionEncoder()
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
            adapter = _StaticQwen3VLVisionForward(
                encoder._visual, grid, 1, encoder._device
            ).eval()
            with torch.inference_mode():
                adapter_output = adapter(
                    item.pixel_values.to(
                        device=encoder._device, dtype=encoder._visual.dtype
                    )
                ).cpu()
            native_output = encoder._forward_eager([item])[0]
            torch.testing.assert_close(
                adapter_output, native_output, rtol=1e-2, atol=1e-2
            )
            print(f"adapter_parity_ok grid={grid}")
            for bucket in buckets:
                for real_count in range(1, bucket + 1):
                    items = [item] * real_count
                    eager = encoder._forward_eager(items)
                    graphed = encoder._forward_graph(items, bucket)
                    assert len(eager) == len(graphed) == real_count
                    for expected, actual in zip(eager, graphed):
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-2, atol=1e-2
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

        cached_item = replace(
            templates[0], content_digest=hashlib.sha256(b"cache-parity").digest()
        )
        first = encoder.forward_batch([cached_item], target_bucket=1)[0]
        expected = first.clone()
        first.zero_()
        second = encoder.forward_batch([cached_item], target_bucket=1)[0]
        torch.testing.assert_close(second, expected, rtol=0, atol=0)
        assert encoder._embedding_cache is not None
        stats = encoder._embedding_cache.stats
        assert stats["hits"] == stats["misses"] == 1
        print(
            "embedding_cache_hit_ok "
            f"entries={stats['entries']} current_bytes={stats['current_bytes']}"
        )
    finally:
        encoder.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--replay-iterations", type=int, default=20)
    args = parser.parse_args()
    verify(args.model, args.replay_iterations)


if __name__ == "__main__":
    main()

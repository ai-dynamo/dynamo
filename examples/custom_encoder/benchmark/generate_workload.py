# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the repeated 300x300 Qwen3-VL CustomEncoder workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from transformers import AutoProcessor

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_SIZE = (300, 300)
BASE_PROMPT = "Classify the image and briefly explain the label."


def _make_images(image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    color = (37, 83, 149)
    image = Image.new("RGB", IMAGE_SIZE, color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            IMAGE_SIZE[0] // 8,
            IMAGE_SIZE[1] // 8,
            IMAGE_SIZE[0] * 7 // 8,
            IMAGE_SIZE[1] * 7 // 8,
        ),
        outline=(255 - color[0], 255 - color[1], 255 - color[2]),
        width=max(2, IMAGE_SIZE[0] // 50),
    )
    draw.text((IMAGE_SIZE[0] // 3, IMAGE_SIZE[1] // 2), "image", fill="white")
    path = image_dir / "image_300x300.jpg"
    image.save(path, format="JPEG", quality=88)
    return [path]


def _visual_token_counts(processor: Any, image_paths: list[Path]) -> list[int]:
    images: list[Image.Image] = []
    try:
        for path in image_paths:
            images.append(Image.open(path).convert("RGB"))
        inputs = processor.image_processor(images=images, return_tensors="pt")
    finally:
        for image in images:
            image.close()
    merge_size = processor.image_processor.merge_size
    return [
        int(grid.prod().item() // merge_size**2) for grid in inputs["image_grid_thw"]
    ]


def _rendered_token_count(tokenizer: Any, prompt: str) -> int:
    rendered = (
        f"<|im_start|>user\n<|image_pad|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _calibrate_prompt(
    tokenizer: Any,
    visual_token_counts: list[int],
    target_mean_isl: int,
) -> tuple[str, int]:
    mean_visual_tokens = sum(visual_token_counts) / len(visual_token_counts)
    for repeat_count in range(1025):
        prompt = BASE_PROMPT + " benchmark" * repeat_count
        text_tokens = _rendered_token_count(tokenizer, prompt)
        mean_isl = text_tokens - 1 + mean_visual_tokens
        if mean_isl == target_mean_isl:
            return prompt, int(mean_isl)
        if mean_isl > target_mean_isl:
            break
    raise ValueError(
        f"could not calibrate an exact ISL of {target_mean_isl}; "
        f"last candidate was {mean_isl}"
    )


def generate_workload(
    output_dir: Path,
    model: str,
    request_count: int,
    target_mean_isl: int,
) -> Path:
    image_paths = _make_images(output_dir / "images")
    processor = AutoProcessor.from_pretrained(model)
    visual_token_counts = _visual_token_counts(processor, image_paths)
    prompt, estimated_mean_isl = _calibrate_prompt(
        processor.tokenizer,
        visual_token_counts,
        target_mean_isl,
    )

    dataset_path = output_dir / (
        f"qwen3_vl_300x300_{request_count}req_isl{target_mean_isl}.jsonl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as dataset:
        for request_index in range(request_count):
            row = {
                "text": prompt,
                "image": str(image_paths[request_index % len(image_paths)].resolve()),
            }
            dataset.write(json.dumps(row) + "\n")

    print(f"dataset={dataset_path}")
    print(f"requests={request_count} unique_images={len(image_paths)}")
    print(f"visual_tokens={visual_token_counts}")
    print(f"estimated_mean_isl={estimated_mean_isl:.2f}")
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".data",
    )
    parser.add_argument("--request-count", type=int, default=1000)
    parser.add_argument("--target-mean-isl", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_workload(
        output_dir=args.output_dir,
        model=args.model,
        request_count=args.request_count,
        target_mean_isl=args.target_mean_isl,
    )


if __name__ == "__main__":
    main()

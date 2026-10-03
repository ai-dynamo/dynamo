# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate multi-turn JSONL datasets for AIPerf raw payload replay."""

from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from args import parse_args

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_images import generate_image_pool_base64, generate_image_pool_http
from generate_input_text import generate_filler


def _load_as_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_image_pools(
    phases: list[dict[str, Any]],
    rng: random.Random,
    np_rng: np.random.Generator,
    output_dir: Path,
    pool_size: int,
    image_mode: str,
    coco_annotations: Path | None,
) -> dict[tuple[int, int], list[str]]:
    sizes: set[tuple[int, int]] = set()
    for phase in phases:
        imgs = phase.get("images")
        if imgs:
            if "width" not in imgs or "height" not in imgs:
                raise ValueError(
                    f"Phase specifies images but missing width/height: {imgs}"
                )
            sizes.add((imgs["width"], imgs["height"]))

    pools: dict[tuple[int, int], list[str]] = {}
    for w, h in sorted(sizes):
        if image_mode == "http":
            assert coco_annotations is not None
            pools[(w, h)] = generate_image_pool_http(rng, pool_size, coco_annotations)
        else:
            image_dir = output_dir / f"image_pool_{w}x{h}"
            paths = generate_image_pool_base64(np_rng, pool_size, image_dir, (w, h))
            pool = [_load_as_data_url(p) for p in paths]
            rng.shuffle(pool)
            pools[(w, h)] = pool
    return pools


def _build_user_content(text: str, image_urls: list[str]) -> str | list[dict[str, Any]]:
    if not image_urls:
        return text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_urls:
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def build_conversation(
    config: dict[str, Any],
    rng: random.Random,
    image_pools: dict[tuple[int, int], list[str]],
    system_texts: list[str],
    wrap_sys_to_user: bool = False,
) -> list[dict[str, Any]]:
    model = config["defaults"]["model"]
    accumulated: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []

    for turn_idx, phase in enumerate(config["phases"]):
        system_text = system_texts[turn_idx]

        images: list[str] = []
        imgs = phase.get("images")
        if imgs:
            w, h = imgs["width"], imgs["height"]
            pool = image_pools[(w, h)]
            images = pool[: imgs["count"]]
            del pool[: imgs["count"]]

        user_text = generate_filler(rng, phase["user_tokens"])

        if wrap_sys_to_user and turn_idx > 0:
            user_text = system_text + "\n" + user_text
            user_msg = {
                "role": "user",
                "content": _build_user_content(user_text, images),
            }
            turn_messages = accumulated + [user_msg]
        else:
            system_msg = {
                "role": "system",
                "content": system_text,
            }
            user_msg = {
                "role": "user",
                "content": _build_user_content(user_text, images),
            }
            turn_messages = accumulated + [system_msg, user_msg]
        payload = {
            "messages": turn_messages,
            "max_tokens": phase["max_output_tokens"],
            "model": model,
        }
        if config.get("extra_inputs") or phase.get("extra_inputs"):
            merged = {**config.get("extra_inputs", {}), **phase.get("extra_inputs", {})}
            payload.update(merged)
        lines.append(payload)

        assistant_msg = {
            "role": "assistant",
            "content": generate_filler(rng, phase["max_output_tokens"]),
        }
        accumulated = turn_messages + [assistant_msg]

    return lines


def write_conversation(path: Path, lines: list[dict[str, Any]]) -> int:
    data = b"\n".join(json.dumps(line).encode() for line in lines) + b"\n"
    path.write_bytes(data)
    return len(data)


def main() -> None:
    args = parse_args(__doc__)

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images_per_conv = sum(
        p["images"]["count"] for p in config["phases"] if p.get("images")
    )
    total_needed = images_per_conv * args.num_conversations
    image_pool_size = args.image_pool_size or total_needed

    reuse_ratio = max(1 - image_pool_size / total_needed, 0) if total_needed > 0 else 0
    print(
        f"Image pool: {image_pool_size} unique images, "
        f"{total_needed} needed ({images_per_conv}/conv x {args.num_conversations} convs), "
        f"{reuse_ratio:.1%} reuse across conversations"
    )

    image_pools = build_image_pools(
        config["phases"],
        rng,
        np_rng,
        args.output_dir,
        image_pool_size,
        args.image_mode,
        args.coco_annotations,
    )

    system_texts = [
        generate_filler(rng, phase["system_tokens"]) for phase in config["phases"]
    ]

    total_bytes = 0
    for i in range(args.num_conversations):
        lines = build_conversation(
            config, rng, image_pools, system_texts, args.wrap_sys_to_user
        )
        path = args.output_dir / f"session_{i + 1:06d}.jsonl"
        total_bytes += write_conversation(path, lines)

    num_phases = len(config["phases"])
    print(
        f"Generated {args.num_conversations} conversations "
        f"({num_phases} turns each, {total_bytes / (1024 * 1024):.1f} MB) "
        f"in {args.output_dir}"
    )


if __name__ == "__main__":
    main()

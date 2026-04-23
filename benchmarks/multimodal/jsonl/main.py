# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate .jsonl benchmark files for aiperf.

Strategies:
    single-turn      Independent requests with random image sampling (default)
    sliding-window   Causal sessions with sliding-window image overlap

Usage:
    python main.py -n 200 --images-pool 100
    python main.py single-turn --image-mode http
    python main.py sliding-window --num-users 10 --turns-per-user 20 --window-size 5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from args import parse_args
from generate_images import (
    generate_image_pool_base64,
    generate_image_pool_http,
    sample_slots,
)
from generate_input_text import generate_filler


def _make_pool(
    args: argparse.Namespace,
    pool_size: int,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> list[str]:
    if args.image_mode == "http":
        return generate_image_pool_http(py_rng, pool_size, args.coco_annotations)
    return generate_image_pool_base64(
        np_rng, pool_size, args.image_dir, tuple(args.image_size)
    )


def run_single_turn(
    args: argparse.Namespace,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> None:
    num_requests: int = args.num_requests
    images_per_request: int = args.images_per_request
    image_pool: int = args.images_pool or (num_requests * images_per_request)

    pool = _make_pool(args, image_pool, np_rng, py_rng)
    slot_refs = sample_slots(py_rng, pool, num_requests, images_per_request)

    output_path = args.output or (
        Path(__file__).parent
        / f"{num_requests}req_{images_per_request}img_{image_pool}pool_{args.user_text_tokens}word_{args.image_mode}.jsonl"
    )

    with open(output_path, "w") as f:
        for i in range(num_requests):
            user_text = generate_filler(py_rng, args.user_text_tokens)
            start = i * images_per_request
            images = slot_refs[start : start + images_per_request]
            line = json.dumps(
                {"text": user_text, "images": images}, separators=(",", ":")
            )
            f.write(line + "\n")

    print(f"Wrote {num_requests} requests to {output_path}")


def run_sliding_window(
    args: argparse.Namespace,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> None:
    num_users: int = args.num_users
    turns_per_user: int = args.turns_per_user
    window_size: int = args.window_size

    images_per_user = window_size + turns_per_user - 1
    total_images = num_users * images_per_user
    total_requests = num_users * turns_per_user

    print(
        f"Sliding window: {num_users} users × {turns_per_user} turns, "
        f"window={window_size}, {images_per_user} images/user, "
        f"{total_images} total images"
    )

    pool = _make_pool(args, total_images, np_rng, py_rng)

    output_path = args.output or (
        Path(__file__).parent
        / f"{num_users}u_{turns_per_user}t_{window_size}w_{args.user_text_tokens}word_{args.image_mode}.jsonl"
    )

    with open(output_path, "w") as f:
        for turn_idx in range(turns_per_user):
            for user_idx in range(num_users):
                offset = user_idx * images_per_user + turn_idx
                window = pool[offset : offset + window_size]
                entry = {
                    "session_id": f"user_{user_idx}",
                    "text": generate_filler(py_rng, args.user_text_tokens),
                    "images": window,
                }
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(f"Wrote {total_requests} requests ({num_users} sessions) to {output_path}")


STRATEGIES = {
    "single-turn": run_single_turn,
    "sliding-window": run_sliding_window,
}


def main() -> None:
    args = parse_args(__doc__)

    seed: int = (
        args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)
    )
    print(f"Using seed: {seed}")

    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    STRATEGIES[args.strategy](args, np_rng, py_rng)


if __name__ == "__main__":
    main()

"""Generate a .jsonl benchmark file for aiperf with single-turn requests.

Each request contains user text (~4,300 tokens by default) and N images.
Images are drawn from a fixed pool (--image-pool); a smaller pool relative to
total image slots produces more cross-request reuse (see README.md).

Two image modes are supported:
  - base64 (default): Generate random 512x512 PNG files locally; JSONL references
    them by absolute file path.
  - http: Sample real images from COCO test2017 URLs (parsed from
    annotations/image_info_test2017.json); JSONL references them by HTTP URL.

System prompt length is NOT in the JSONL — pass it via aiperf's
--shared-system-prompt-length flag.

Usage:
    python generate_requests.py
    python generate_requests.py --image-mode http
    python generate_requests.py -n 200 --images-pool 100
    python generate_requests.py -n 100 --images-per-request 20 --images-pool 500
    python generate_requests.py --user-text-tokens 4000
    python generate_requests.py -o out.jsonl --image-dir /tmp/bench_images

Example aiperf invocation:
    aiperf profile \\
      --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \\
      --input-file examples/backends/vllm/launch/gtc/requests.jsonl \\
      --custom-dataset-type single_turn \\
      --shared-system-prompt-length 8600 \\
      --extra-inputs "max_tokens:500" \\
      --extra-inputs "min_tokens:500" \\
      --extra-inputs "ignore_eos:true"

  The file contains the actual request content (user text + image paths/URLs),
  not token counts. Input length is computed from that content plus
  --shared-system-prompt-length. Do not pass --isl: it applies only to
  synthetic data generation, not to --input-file single_turn.
"""

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

SEED = int(time.time() * 1000) % (2**32)


def main() -> None:
    args = parse_args(__doc__)
    num_requests: int = args.num_requests
    images_per_request: int = args.images_per_request
    image_pool: int = args.images_pool or (num_requests * images_per_request)

    np_rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    if args.image_mode == "http":
        pool = generate_image_pool_http(py_rng, image_pool, args.coco_annotations)
    else:
        pool = generate_image_pool_base64(
            np_rng, image_pool, args.image_dir, tuple(args.image_size)
        )
    slot_refs = sample_slots(py_rng, pool, num_requests, images_per_request)

    unique_images = len(set(slot_refs))
    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).parent
            / f"{num_requests}req_{images_per_request}img_{unique_images}pool_{args.user_text_tokens}word_{args.image_mode}.jsonl"
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


if __name__ == "__main__":
    main()

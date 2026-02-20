"""Generate a .jsonl benchmark file for aiperf (single-turn, text + images).

Images are drawn from a fixed pool; a smaller pool produces more cross-request
reuse. Supports base64 (local PNGs) and http (COCO URLs) image modes.

Usage:
    python main.py
    python main.py --image-mode http
    python main.py -n 200 --images-pool 100
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

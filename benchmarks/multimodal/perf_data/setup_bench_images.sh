#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the 512x512 random PNG images needed by the benchmark JSONL datasets.
# The JSONL files reference /tmp/bench_images_512/img_XXXX.png.
# Image content is random noise — only filenames matter for reuse patterns.
#
# Usage:
#   bash setup_bench_images.sh [IMAGE_COUNT] [IMAGE_DIR]
#   Defaults: 5000 images, /tmp/bench_images_512
#
# Requirements: python3, numpy, Pillow
#   pip install numpy Pillow

set -euo pipefail

IMAGE_COUNT="${1:-5000}"
IMAGE_DIR="${2:-/tmp/bench_images_512}"

if [[ -d "${IMAGE_DIR}" ]] && [[ "$(ls "${IMAGE_DIR}"/img_*.png 2>/dev/null | wc -l)" -ge "${IMAGE_COUNT}" ]]; then
    echo "Images already exist in ${IMAGE_DIR} ($(ls "${IMAGE_DIR}"/img_*.png | wc -l) files). Skipping."
    exit 0
fi

echo "Generating ${IMAGE_COUNT} random 512x512 PNG images in ${IMAGE_DIR} ..."

python3 - "${IMAGE_DIR}" "${IMAGE_COUNT}" <<'PYEOF'
import sys
import numpy as np
from PIL import Image
from pathlib import Path

image_dir = Path(sys.argv[1])
image_count = int(sys.argv[2])

image_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
for i in range(image_count):
    pixels = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image_dir / f"img_{i:04d}.png")
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{image_count} done")
print(f"Done. {image_count} images saved to {image_dir}")
PYEOF

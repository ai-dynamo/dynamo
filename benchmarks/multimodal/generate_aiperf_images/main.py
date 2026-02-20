from pathlib import Path

import aiperf.dataset.generator.image as _img_mod
import numpy as np
from args import parse_args
from PIL import Image

TARGET_DIR = Path(_img_mod.__file__).parent / "assets" / "source_images"


def main() -> None:
    args = parse_args()
    num_images: int = args.images_pool
    width, height = args.image_size

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(num_images):
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(TARGET_DIR / f"noise_{i:04d}.png")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{num_images}")
    print(f"\n{num_images} unique {width}x{height} images saved to {TARGET_DIR}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS V1 cold-storage saver CLI."""

from __future__ import annotations

import argparse
import logging
import os

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.v1.snapshot import save_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Save GMS V1 weights.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-size-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args(argv)

    save_weights(
        os.path.join(args.checkpoint_dir, f"device-{args.device}"),
        get_socket_path(args.device, "weights"),
        args.device,
        shard_size_bytes=args.shard_size_bytes,
    )


if __name__ == "__main__":
    main()

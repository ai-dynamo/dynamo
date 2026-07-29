# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS V1 one-shot cold-storage loader CLI."""

from __future__ import annotations

import argparse
import logging
import os

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.v1.snapshot import hydrate_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Hydrate GMS V1 weights into a fresh rank-local server.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args(argv)

    hydrate_weights(
        os.path.join(args.checkpoint_dir, f"device-{args.device}"),
        get_socket_path(args.device, "weights"),
        args.device,
        max_workers=args.max_workers,
    )
    logger.info("GMS V1 loader complete; exiting")


if __name__ == "__main__":
    main()

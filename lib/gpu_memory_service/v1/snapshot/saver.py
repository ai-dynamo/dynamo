# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS V1 exact-allocation weight saver CLI."""

from __future__ import annotations

import os

from gpu_memory_service.cli.snapshot.saver import _build_parser as _build_v0_parser
from gpu_memory_service.common.vmm import VMMDeviceType, init_vmm
from gpu_memory_service.snapshot.backends.sharded_ssd import (
    device_sharded_ssd_roots,
    parse_sharded_ssd_roots,
)
from gpu_memory_service.v1.device import get_socket_path
from gpu_memory_service.v1.snapshot.weight_artifact import save_weights


def _build_parser():
    parser = _build_v0_parser()
    parser.description = "Save exact committed GMS V1 weight allocations."
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA-visible rank-local device ordinal.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoint_dir:
        parser.error("--checkpoint-dir is required for directory-backed saves")
    if args.device_type != VMMDeviceType.CUDA.value:
        parser.error("GMS V1 only supports --device-type=cuda")
    if args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")
    if args.save_lock_timeout_ms <= 0:
        parser.error("--save-lock-timeout-ms must be a positive integer")

    init_vmm(VMMDeviceType.CUDA)
    roots = device_sharded_ssd_roots(
        args.checkpoint_dir,
        args.device,
        parse_sharded_ssd_roots(args.sharded_ssd_roots),
    )
    save_weights(
        os.path.join(args.checkpoint_dir, f"device-{args.device}"),
        get_socket_path(args.device, "weights"),
        args.device,
        shard_size_bytes=args.shard_size_bytes,
        max_workers=args.max_workers,
        admission_timeout=args.save_lock_timeout_ms / 1000,
        sharded_ssd_roots=roots,
    )


if __name__ == "__main__":
    main()

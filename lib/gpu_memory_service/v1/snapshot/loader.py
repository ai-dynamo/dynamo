# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS V1 one-shot exact-allocation weight loader CLI."""

from __future__ import annotations

import logging
import os

from gpu_memory_service.cli.snapshot.loader import _build_parser as _build_v0_parser
from gpu_memory_service.common.vmm import VMMDeviceType, init_vmm
from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.v1.device import get_socket_path
from gpu_memory_service.v1.snapshot.weight_artifact import load_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_backend_params(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, parameter = value.partition("=")
        if not separator or not key:
            raise ValueError(
                f"invalid POSIX backend parameter {value!r}; expected KEY=VALUE"
            )
        result[key] = parameter
    return result


def _build_parser():
    parser = _build_v0_parser()
    parser.description = "Load exact GMS V1 weights into a fresh rank-local server."
    parser.set_defaults(device=0)
    parser.add_argument(
        "--posix-backend-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "NIXL POSIX backend parameter; may be repeated "
            "(for example, --posix-backend-param ios_pool_size=64)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    parser.add_argument(
        "--socket-device",
        type=int,
        help="Captured engine ordinal used only for socket naming",
    )
    parser.add_argument(
        "--artifact-device",
        type=int,
        help="Captured artifact directory ordinal; defaults to --device",
    )
    args = parser.parse_args(argv)
    socket_device = args.device if args.socket_device is None else args.socket_device
    artifact_device = (
        args.device if args.artifact_device is None else args.artifact_device
    )
    if min(args.device, socket_device, artifact_device) < 0:
        parser.error("device ordinals must be nonnegative")
    if not args.checkpoint_dir:
        parser.error(
            f"--checkpoint-dir is required for --transfer-backend={args.transfer_backend}"
        )
    if args.device_type != VMMDeviceType.CUDA.value:
        parser.error("GMS V1 only supports --device-type=cuda")
    if args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")
    if args.sharded_ssd_queues_per_root <= 0:
        parser.error("--sharded-ssd-queues-per-root must be a positive integer")
    try:
        posix_backend_params = _parse_backend_params(args.posix_backend_param)
    except ValueError as exc:
        parser.error(str(exc))

    init_vmm(VMMDeviceType.CUDA)
    load_weights(
        os.path.join(args.checkpoint_dir, f"device-{artifact_device}"),
        get_socket_path(socket_device, "weights"),
        args.device,
        max_workers=args.max_workers,
        transfer_backend=args.transfer_backend,
        sharded_ssd_roots=parse_sharded_ssd_roots(args.sharded_ssd_roots),
        sharded_ssd_queues_per_root=args.sharded_ssd_queues_per_root,
        posix_backend_params=posix_backend_params,
    )
    logger.info("GMS V1 loader complete; exiting")


if __name__ == "__main__":
    main()

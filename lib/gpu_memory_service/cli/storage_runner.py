# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS Storage Client CLI.

Provides two subcommands for saving and loading GPU Memory Service state:

* ``save`` – connect to a running GMS server in RO mode and write every
              allocation plus all metadata to a sharded binary directory.
* ``load`` – connect to a running GMS server in RW mode, read tensor data
              from a saved directory, and commit the state so readers can
              acquire the RO lock.

Usage examples::

    # Save GMS state to disk
    gms-storage-client save --output-dir /mnt/nvme/save --device 0

    # Load a previous save back into a fresh GMS server
    gms-storage-client load --input-dir /mnt/nvme/save --device 0
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("gpu_memory_service").setLevel(logging.DEBUG)


def _resolve_socket(device: int, socket_path) -> str:
    if socket_path is not None:
        return socket_path
    from gpu_memory_service.common.utils import get_socket_path

    return get_socket_path(device)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _run_save(args) -> None:
    """Execute the save subcommand."""
    from gpu_memory_service.client.gms_storage_client import GMSStorageClient

    _configure_logging(args.verbose)
    socket_path = _resolve_socket(args.device, args.socket_path)

    logger.info(
        "Saving GMS state: device=%s, socket=%s, output_dir=%s, save_workers=%s",
        args.device,
        socket_path,
        args.output_dir,
        args.save_workers,
    )

    client = GMSStorageClient(
        args.output_dir,
        socket_path=socket_path,
        device=args.device,
        timeout_ms=args.timeout_ms,
        shard_size_bytes=args.shard_size_bytes,
    )

    manifest = client.save(max_workers=args.save_workers)
    shard_count = len({a.tensor_file for a in manifest.allocations})

    logger.info(
        "Save complete: %s allocations written to %s (%s shards)",
        len(manifest.allocations),
        args.output_dir,
        shard_count,
    )
    logger.info("Layout hash: %s", manifest.layout_hash)


def _run_load(args) -> None:
    """Execute the load subcommand."""
    from gpu_memory_service.client.gms_storage_client import GMSStorageClient

    _configure_logging(args.verbose)
    socket_path = _resolve_socket(args.device, args.socket_path)

    logger.info(
        "Loading GMS state: device=%s, socket=%s, input_dir=%s, clear_existing=%s",
        args.device,
        socket_path,
        args.input_dir,
        not args.no_clear,
    )

    client = GMSStorageClient(
        socket_path=socket_path,
        device=args.device,
        timeout_ms=args.timeout_ms,
    )

    id_map = client.load_to_gms(
        args.input_dir,
        max_workers=args.workers,
        clear_existing=not args.no_clear,
    )

    logger.info("Load complete: %s allocations committed to GMS", len(id_map))
    for old_id, new_id in id_map.items():
        logger.info("  %s → %s", old_id, new_id)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_SHARD_SIZE_DEFAULT = 4 * 1024**3  # 4 GiB


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gms-storage-client",
        description="Save and load GPU Memory Service state to/from disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # -- save ---------------------------------------------------------------
    save_p = subparsers.add_parser(
        "save",
        help="Save GMS state to a sharded binary directory.",
        description=(
            "Connect to a running GMS server in RO mode and export every "
            "allocation plus all metadata to a compact sharded binary format."
        ),
    )
    save_p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write into (created if absent).",
    )
    save_p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    save_p.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help="GMS Unix socket path. Default uses GPU UUID-based path.",
    )
    save_p.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Timeout in milliseconds for acquiring the RO lock.",
    )
    save_p.add_argument(
        "--shard-size-bytes",
        type=int,
        default=_SHARD_SIZE_DEFAULT,
        help=(
            f"Soft upper bound per shard file in bytes "
            f"(default: {_SHARD_SIZE_DEFAULT // 1024**3} GiB).  "
            "Decrease to increase parallelism on save/load; increase to "
            "reduce file count."
        ),
    )
    save_p.add_argument(
        "--save-workers",
        type=int,
        default=4,
        help="Thread pool size for parallel shard writes (default: 4).",
    )
    save_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    # -- load ---------------------------------------------------------------
    load_p = subparsers.add_parser(
        "load",
        help="Load a saved GMS state back into a running GMS server.",
        description=(
            "Connect to a running GMS server in RW mode, read tensor data "
            "from a saved directory (reading each shard file sequentially), "
            "and commit the state so readers can acquire the RO lock."
        ),
    )
    load_p.add_argument(
        "--input-dir",
        required=True,
        help="Directory previously created by the save subcommand.",
    )
    load_p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    load_p.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help="GMS Unix socket path. Default uses GPU UUID-based path.",
    )
    load_p.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Timeout in milliseconds for acquiring the RW lock.",
    )
    load_p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel shard reads (default: 4).",
    )
    load_p.add_argument(
        "--no-clear",
        action="store_true",
        default=False,
        help=(
            "Do not clear existing GMS allocations before loading. "
            "Default behaviour clears the server to produce an exact replica."
        ),
    )
    load_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the GMS Storage Client CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "save":
        _run_save(args)
    elif args.subcommand == "load":
        _run_load(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

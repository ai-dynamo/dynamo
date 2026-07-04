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

from gpu_memory_service.snapshot.backends.sharded_ssd import parse_sharded_ssd_roots
from gpu_memory_service.snapshot.transfer import TransferBackendKind

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
    from gpu_memory_service.snapshot.storage_client import GMSStorageClient

    _configure_logging(args.verbose)
    socket_path = _resolve_socket(args.device, args.socket_path)

    logger.info(
        "Saving GMS state: device=%s, socket=%s, output_dir=%s, "
        "save_workers=%s, sharded_ssd_roots=%s",
        args.device,
        socket_path,
        args.output_dir,
        args.save_workers,
        args.sharded_ssd_roots or "-",
    )

    client = GMSStorageClient(
        args.output_dir,
        socket_path=socket_path,
        device=args.device,
        timeout_ms=args.timeout_ms,
        shard_size_bytes=args.shard_size_bytes,
        sharded_ssd_roots=parse_sharded_ssd_roots(args.sharded_ssd_roots or ""),
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
    from gpu_memory_service.snapshot.storage_client import GMSStorageClient

    _configure_logging(args.verbose)
    socket_path = _resolve_socket(args.device, args.socket_path)

    logger.info(
        "Loading GMS state: device=%s, socket=%s, input_dir=%s, "
        "clear_existing=%s, transfer_backend=%s",
        args.device,
        socket_path,
        args.input_dir,
        not args.no_clear,
        args.transfer_backend,
    )

    client = GMSStorageClient(
        socket_path=socket_path,
        device=args.device,
        timeout_ms=args.timeout_ms,
        transfer_backend=args.transfer_backend,
        sharded_ssd_roots=parse_sharded_ssd_roots(args.sharded_ssd_roots or ""),
        sharded_ssd_queues_per_root=args.sharded_ssd_queues_per_root,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gms-storage-client",
        description="Save and load GPU Memory Service state to/from disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # -- save ---------------------------------------------------------------
    save_p = subparsers.add_parser(
        "save",
        help="Save GMS state to a sharded binary directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="CUDA device index.",
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
        default=4 * 1024**3,
        help=(
            "Soft upper bound per shard file in bytes. "
            "Decrease to increase parallelism on save/load; increase to "
            "reduce file count."
        ),
    )
    save_p.add_argument(
        "--sharded-ssd-roots",
        default=None,
        help=(
            "Comma-separated SSD roots for prototype sharded saves. "
            "If unset, shards are written under --output-dir."
        ),
    )
    save_p.add_argument(
        "--save-workers",
        type=int,
        default=8,
        help="Thread pool size for parallel shard writes.",
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Connect to a running GMS server in RW mode, restore tensor data "
            "from a saved directory through the selected transfer backend, "
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
        help="CUDA device index.",
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
        default=16,
        help="Thread pool size for parallel shard reads.",
    )
    load_p.add_argument(
        "--transfer-backend",
        choices=[backend.value for backend in TransferBackendKind],
        default=TransferBackendKind.MODELEXPRESS.value,
        help=(
            "Byte transfer backend for restore. "
            "'modelexpress' uses the ModelExpress restore strategy chain; "
            "'nixl' uses NIXL POSIX with host staging; "
            "'nixl-gds' uses NIXL GDS_MT for direct file-to-GPU transfers; "
            "'sharded-ssd' shards reads across local SSD roots using the "
            "same NIXL POSIX host-staging path."
        ),
    )
    load_p.add_argument(
        "--sharded-ssd-roots",
        default=None,
        help="Comma-separated SSD roots for the sharded-ssd transfer backend.",
    )
    load_p.add_argument(
        "--sharded-ssd-queues-per-root",
        type=int,
        default=2,
        help=(
            "Number of independent sharded-ssd restore queues per SSD root. "
            "Increase this "
            "to use multiple NIXL/POSIX workers per local SSD root."
        ),
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
    if args.subcommand == "load" and args.sharded_ssd_queues_per_root <= 0:
        parser.error("--sharded-ssd-queues-per-root must be a positive integer")

    if args.subcommand == "save":
        _run_save(args)
    elif args.subcommand == "load":
        _run_load(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

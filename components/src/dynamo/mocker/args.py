#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from dynamo.common.utils.namespace import get_worker_namespace

from . import __version__
from .utils.planner_profiler_perf_data_converter import (
    convert_profile_results_to_npz,
    is_mocker_format_npz,
    is_profile_results_dir,
)

DYN_NAMESPACE = get_worker_namespace()
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"
DEFAULT_PREFILL_ENDPOINT = f"dyn://{DYN_NAMESPACE}.prefill.generate"

logger = logging.getLogger(__name__)


class ProfileDataResult:
    """Result of processing --planner-profile-data argument. Cleans up tmpdir on deletion."""

    def __init__(
        self, npz_path: Path | None, tmpdir: tempfile.TemporaryDirectory | None
    ):
        self.npz_path = npz_path
        self._tmpdir = tmpdir

    def __del__(self):
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
                logger.debug("Cleaned up profile data temporary directory")
            except Exception:
                pass  # Best effort cleanup


def resolve_planner_profile_data(
    planner_profile_data: Path | None,
) -> ProfileDataResult:
    """
    Resolve --planner-profile-data to an NPZ file path.

    Handles backward compatibility by accepting either:
    1. A mocker-format NPZ file (returned as-is)
    2. A profiler-style results directory (converted to mocker-format NPZ)

    Args:
        planner_profile_data: Path from --planner-profile-data argument.

    Returns:
        ProfileDataResult with npz_path and optional tmpdir for cleanup.

    Raises:
        FileNotFoundError: If path doesn't contain valid profile data in any supported format.
    """
    if planner_profile_data is None:
        return ProfileDataResult(npz_path=None, tmpdir=None)

    # Case 1: Already a mocker-format NPZ file
    if is_mocker_format_npz(planner_profile_data):
        logger.info(f"Using mocker-format NPZ file: {planner_profile_data}")
        return ProfileDataResult(npz_path=planner_profile_data, tmpdir=None)

    # Case 2: Profiler-style results directory - needs conversion
    if is_profile_results_dir(planner_profile_data):
        logger.info(
            f"Detected profiler-style results directory at {planner_profile_data}, converting to NPZ..."
        )
        tmpdir = tempfile.TemporaryDirectory(prefix="mocker_perf_data_")
        npz_path = Path(tmpdir.name) / "perf_data.npz"
        convert_profile_results_to_npz(planner_profile_data, npz_path)
        return ProfileDataResult(npz_path=npz_path, tmpdir=tmpdir)

    # Case 3: Invalid path - neither mocker-format NPZ nor profiler-style directory
    raise FileNotFoundError(
        f"Path '{planner_profile_data}' is neither a mocker-format NPZ file nor a valid profiler results directory.\n"
        f"Expected either:\n"
        f"  - A .npz file with keys: prefill_isl, prefill_ttft_ms, decode_active_kv_tokens, decode_context_length, decode_itl\n"
        f"  - A directory containing selected_prefill_interpolation/raw_data.npz and selected_decode_interpolation/raw_data.npz\n"
        f"  - A directory containing prefill_raw_data.json and decode_raw_data.json"
    )


def create_temp_engine_args_file(args) -> Path:
    """
    Create a temporary JSON file with MockEngineArgs from CLI arguments.
    Returns the path to the temporary file.
    """
    engine_args = {}

    # Only include non-None values that differ from defaults
    # Note: argparse converts hyphens to underscores in attribute names
    # Extract all potential engine arguments, using None as default for missing attributes
    engine_args = {
        "num_gpu_blocks": getattr(args, "num_gpu_blocks", None),
        "block_size": getattr(args, "block_size", None),
        "max_num_seqs": getattr(args, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", None),
        "enable_chunked_prefill": getattr(args, "enable_chunked_prefill", None),
        "watermark": getattr(args, "watermark", None),
        "speedup_ratio": getattr(args, "speedup_ratio", None),
        "dp_size": getattr(args, "dp_size", None),
        "startup_time": getattr(args, "startup_time", None),
        "planner_profile_data": (
            str(getattr(args, "planner_profile_data", None))
            if getattr(args, "planner_profile_data", None)
            else None
        ),
        "is_prefill": getattr(args, "is_prefill_worker", None),
        "is_decode": getattr(args, "is_decode_worker", None),
        "enable_local_indexer": not getattr(args, "durable_kv_events", False),
        # Note: bootstrap_port is NOT included here - it's set per-worker in launch_workers()
    }

    # Parse --reasoning JSON string into a nested object
    reasoning_str = getattr(args, "reasoning", None)
    if reasoning_str:
        engine_args["reasoning"] = json.loads(reasoning_str)

    # Remove None values to only include explicitly set arguments
    engine_args = {k: v for k, v in engine_args.items() if v is not None}

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(engine_args, f, indent=2)
        temp_path = Path(f.name)

    logger.debug(f"Created temporary MockEngineArgs file at {temp_path}")
    logger.debug(f"MockEngineArgs: {engine_args}")

    return temp_path


def validate_worker_type_args(args):
    """
    Resolve disaggregation mode from --disaggregation-mode or legacy boolean flags.
    Raises ValueError if validation fails.
    """
    import warnings

    explicit_mode = args.disaggregation_mode is not None
    has_legacy = args.is_prefill_worker or args.is_decode_worker

    if has_legacy and explicit_mode:
        raise ValueError(
            "Cannot combine --is-prefill-worker/--is-decode-worker with "
            "--disaggregation-mode. Use only --disaggregation-mode."
        )

    if has_legacy:
        if args.is_prefill_worker and args.is_decode_worker:
            raise ValueError(
                "Cannot specify both --is-prefill-worker and --is-decode-worker. "
                "A worker must be either prefill, decode, or aggregated (neither flag set)."
            )
        if args.is_prefill_worker:
            warnings.warn(
                "--is-prefill-worker is deprecated, use --disaggregation-mode=prefill",
                DeprecationWarning,
                stacklevel=2,
            )
            args.disaggregation_mode = "prefill"
        elif args.is_decode_worker:
            warnings.warn(
                "--is-decode-worker is deprecated, use --disaggregation-mode=decode",
                DeprecationWarning,
                stacklevel=2,
            )
            args.disaggregation_mode = "decode"

    # Apply default if neither new flag nor legacy flags were provided
    if args.disaggregation_mode is None:
        args.disaggregation_mode = "agg"

    # Sync booleans from disaggregation_mode
    args.is_prefill_worker = args.disaggregation_mode == "prefill"
    args.is_decode_worker = args.disaggregation_mode == "decode"


def parse_bootstrap_ports(ports_str: str | None) -> list[int]:
    """Parse comma-separated bootstrap ports string into list of integers."""
    if not ports_str:
        return []
    return [int(p.strip()) for p in ports_str.split(",")]


def parse_args():
    """Parse command-line arguments for the Dynamo mocker engine.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mocker engine for testing Dynamo LLM infrastructure with vLLM-style CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Mocker {__version__}"
    )

    # Basic configuration
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory or HuggingFace model ID for tokenizer",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=f"Dynamo endpoint string (default: {DEFAULT_ENDPOINT} for aggregated/decode, {DEFAULT_PREFILL_ENDPOINT} for prefill)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for API responses (default: derived from model-path)",
    )

    # MockEngineArgs parameters (similar to vLLM style)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        dest="num_gpu_blocks",  # Maps to num_gpu_blocks in MockEngineArgs
        default=None,
        help="Number of GPU blocks for KV cache (default: 16384)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Token block size for KV cache blocks (default: 64)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences per iteration (default: 256)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration (default: 8192)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        dest="enable_prefix_caching",
        default=None,
        help="Enable automatic prefix caching (default: True)",
    )
    parser.add_argument(
        "--no-enable-prefix-caching",
        action="store_false",
        dest="enable_prefix_caching",
        default=None,
        help="Disable automatic prefix caching",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        dest="enable_chunked_prefill",
        default=None,
        help="Enable chunked prefill (default: True)",
    )
    parser.add_argument(
        "--no-enable-chunked-prefill",
        action="store_false",
        dest="enable_chunked_prefill",
        default=None,
        help="Disable chunked prefill",
    )
    parser.add_argument(
        "--watermark",
        type=float,
        default=None,
        help="Watermark value for the mocker engine (default: 0.01)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=None,
        help="Speedup ratio for mock execution (default: 1.0). Use 0 for infinite speedup (no simulation delays).",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        dest="dp_size",
        default=None,
        help="Number of data parallel replicas (default: 1)",
    )
    parser.add_argument(
        "--startup-time",
        type=float,
        default=None,
        help="Simulated engine startup time in seconds (default: None)",
    )
    parser.add_argument(
        "--planner-profile-data",
        type=Path,
        default=None,
        help="Path to profile results directory containing selected_prefill_interpolation/ and "
        "selected_decode_interpolation/ subdirectories (default: None, uses hardcoded polynomials)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of mocker workers to launch in the same process (default: 1). "
        "All workers share the same tokio runtime and thread pool.",
    )

    # Reasoning token output
    parser.add_argument(
        "--reasoning",
        type=str,
        default=None,
        help="Enable reasoning token output. JSON object with fields: "
        "start_thinking_token_id (u32), end_thinking_token_id (u32), thinking_ratio (0.0-1.0). "
        'Example: \'{"start_thinking_token_id": 123, "end_thinking_token_id": 456, "thinking_ratio": 0.6}\'',
    )

    # Legacy support - allow direct JSON file specification
    parser.add_argument(
        "--extra-engine-args",
        type=Path,
        help="Path to JSON file with mocker configuration. "
        "If provided, overrides individual CLI arguments.",
    )

    # Worker type configuration
    parser.add_argument(
        "--disaggregation-mode",
        type=str,
        default=None,
        choices=["agg", "prefill", "decode"],
        help="Worker disaggregation mode: 'agg' (default, aggregated), "
        "'prefill' (prefill-only worker), or 'decode' (decode-only worker).",
    )
    parser.add_argument(
        "--is-prefill-worker",
        action="store_true",
        default=False,
        help="DEPRECATED: use --disaggregation-mode=prefill. "
        "Register as Prefill model type instead of Chat+Completions (default: False)",
    )
    parser.add_argument(
        "--is-decode-worker",
        action="store_true",
        default=False,
        help="DEPRECATED: use --disaggregation-mode=decode. "
        "Mark this as a decode worker which does not publish KV events (default: False)",
    )
    parser.add_argument(
        "--durable-kv-events",
        action="store_true",
        default=os.environ.get("DYN_DURABLE_KV_EVENTS", "false").lower() == "true",
        help="Enable durable KV events using NATS JetStream instead of the local indexer. By default, local indexer is enabled for lower latency. Use this flag when you need durability and multi-replica router consistency. Requires NATS with JetStream enabled. Can also be set via DYN_DURABLE_KV_EVENTS=true env var.",
    )
    parser.add_argument(
        "--bootstrap-ports",
        type=str,
        default=None,
        help="Comma-separated list of bootstrap ports for disaggregated serving rendezvous. "
        "One port per worker (must match --num-workers). "
        "Prefill workers listen on these ports; decode workers connect to them. "
        "If not specified, bootstrap rendezvous is disabled.",
    )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=-1.0,
        help=(
            "Delay in seconds between launching each worker to avoid overwhelming "
            "etcd/NATS/frontend with many workers. Set to 0 to disable staggering. "
            "Use -1 for auto mode (0.1s for 32-128 workers, 0.2s for >128 workers, 0 otherwise). "
            "Default: -1 (auto)"
        ),
    )
    parser.add_argument(
        "--discovery-backend",
        type=str,
        choices=["kubernetes", "etcd", "file", "mem"],
        default=os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
        help="Discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), mem (in-memory). Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        choices=["nats", "http", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    )

    args = parser.parse_args()
    validate_worker_type_args(args)

    # Validate num_workers
    if args.num_workers < 1:
        raise ValueError(f"--num-workers must be at least 1, got {args.num_workers}")

    # Parse and validate bootstrap_ports
    args.bootstrap_ports_list = parse_bootstrap_ports(args.bootstrap_ports)
    if args.bootstrap_ports_list:
        if len(args.bootstrap_ports_list) != args.num_workers:
            raise ValueError(
                f"--bootstrap-ports must have exactly --num-workers ({args.num_workers}) ports, "
                f"got {len(args.bootstrap_ports_list)}: {args.bootstrap_ports_list}"
            )

    # Set endpoint default based on worker type if not explicitly provided
    if args.endpoint is None:
        if args.is_prefill_worker:
            args.endpoint = DEFAULT_PREFILL_ENDPOINT
            logger.debug(f"Using default prefill endpoint: {args.endpoint}")
        else:
            args.endpoint = DEFAULT_ENDPOINT
            logger.debug(f"Using default endpoint: {args.endpoint}")

    return args

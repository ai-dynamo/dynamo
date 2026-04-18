#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import tempfile
from pathlib import Path

from dynamo.common.utils.namespace import get_worker_namespace

from . import __version__

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

    from .utils.planner_profiler_perf_data_converter import (
        convert_profile_results_to_npz,
        is_mocker_format_npz,
        is_profile_results_dir,
    )

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


def validate_worker_type_args(args: argparse.Namespace) -> None:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        default=16384,
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
        default=256,
        help="Maximum number of sequences per iteration (default: 256)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Maximum number of batched tokens per iteration (default: 2048)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        dest="enable_prefix_caching",
        default=True,
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
        default=True,
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
        "--preemption-mode",
        type=str,
        default="lifo",
        choices=["lifo", "fifo"],
        help="Preemption mode for decode eviction under memory pressure. "
        "'lifo' (default) evicts the newest request (matches vLLM v1), "
        "'fifo' evicts the oldest request.",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=1.0,
        help="Speedup ratio for mock execution (default: 1.0). Use 0 for infinite speedup (no simulation delays).",
    )
    parser.add_argument(
        "--decode-speedup-ratio",
        type=float,
        default=1.0,
        help="Additional speedup multiplier applied only to decode steps (default: 1.0). "
        "Models speculative decoding (e.g. Eagle) where decode throughput improves "
        "without affecting prefill latency. Effective decode speedup is speedup_ratio * decode_speedup_ratio.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        dest="dp_size",
        default=1,
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
        "--aic-perf-model",
        action="store_true",
        default=False,
        help="Use direct AIC SDK calls for latency prediction. "
        "Requires aiconfigurator SDK installed.",
    )
    parser.add_argument(
        "--aic-system",
        type=str,
        default=None,
        help="AIC system name (e.g., 'h200_sxm'). Used with --aic-perf-model.",
    )
    parser.add_argument(
        "--aic-backend-version",
        type=str,
        default=None,
        help="AIC backend engine version (e.g., '0.12.0' for vLLM, '0.5.6.post2' for SGLang). "
        "If not set, uses the default version for the backend.",
    )
    parser.add_argument(
        "--aic-tp-size",
        type=int,
        default=None,
        help="Tensor parallel size for AIC latency prediction (default: 1). "
        "Only affects AIC performance model lookups, not mocker scheduling.",
    )
    parser.add_argument(
        "--aic-moe-tp-size",
        type=int,
        default=None,
        help="MoE tensor-parallel size for AIC latency prediction. "
        "Required for MoE models. Constraint: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.",
    )
    parser.add_argument(
        "--aic-moe-ep-size",
        type=int,
        default=None,
        help="MoE expert-parallel size for AIC latency prediction. "
        "Required for MoE models. Constraint: aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size.",
    )
    parser.add_argument(
        "--aic-attention-dp-size",
        type=int,
        default=None,
        help="Attention data-parallel size for AIC latency prediction (default: 1). "
        "Corresponds to the 'dp' dimension in AIC CLI output.",
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

    # Engine type selection
    parser.add_argument(
        "--engine-type",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Engine simulation type: 'vllm' (default) or 'sglang'.",
    )

    # SGLang-specific configuration
    parser.add_argument(
        "--sglang-schedule-policy",
        type=str,
        default=None,
        choices=["fifo", "fcfs", "lpm"],
        help="SGLang scheduling policy: 'fifo'/'fcfs' (default) or 'lpm' (longest prefix match).",
    )
    parser.add_argument(
        "--sglang-page-size",
        type=int,
        default=None,
        help="SGLang radix cache page size in tokens (default: 1).",
    )
    parser.add_argument(
        "--sglang-max-prefill-tokens",
        type=int,
        default=None,
        help="SGLang maximum prefill tokens budget per batch (default: 16384).",
    )
    parser.add_argument(
        "--sglang-chunked-prefill-size",
        type=int,
        default=None,
        help="SGLang chunked prefill size — max tokens per chunk (default: 8192).",
    )
    parser.add_argument(
        "--sglang-clip-max-new-tokens",
        type=int,
        default=None,
        help="SGLang clip max new tokens for admission budget (default: 4096).",
    )
    parser.add_argument(
        "--sglang-schedule-conservativeness",
        type=float,
        default=None,
        help="SGLang schedule conservativeness factor 0.0-1.0 (default: 1.0).",
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
        help="[Deprecated] Enable durable KV events using NATS JetStream. This option will be removed in a future release. The event-plane subscriber (local_indexer mode) is now the recommended path.",
    )
    parser.add_argument(
        "--zmq-kv-events-ports",
        type=str,
        default=None,
        help="Comma-separated list of ZMQ PUB base ports for KV event publishing "
        "in vLLM native wire format. One port per worker (must match --num-workers). "
        "Each worker's DP ranks bind on base_port + dp_rank. A KvEventPublisher relay "
        "subscribes and forwards events to NATS. (default: None, disabled)",
    )
    parser.add_argument(
        "--zmq-replay-ports",
        type=str,
        default=None,
        help="Comma-separated list of ZMQ ROUTER base ports for KV event replay. "
        "One port per worker (must match --num-workers). "
        "Each worker's DP ranks bind on base_port + dp_rank. "
        "Used alongside --zmq-kv-events-ports for gap recovery. (default: None, disabled)",
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

    # KV cache transfer latency simulation
    parser.add_argument(
        "--kv-transfer-bandwidth",
        type=float,
        default=_default_kv_transfer_bandwidth_gbps(),
        help="KV cache transfer bandwidth in GB/s for disaggregated serving latency simulation. "
        "Default: 64.0 (inter-node InfiniBand). Set to 0 to disable KV transfer delay. "
        "For intra-node NVLink, typical value is ~450.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=[
            "auto",
            "bfloat16",
            "fp8",
            "fp8_ds_mla",
            "fp8_e4m3",
            "fp8_e5m2",
            "fp8_inc",
        ],
        help="Data type for KV cache, used to compute kv_bytes_per_token. "
        "'auto' uses the model's dtype (default).",
    )
    parser.add_argument(
        "--kv-bytes-per-token",
        type=int,
        default=None,
        help="KV cache bytes per token. If not specified, auto-computed from model config "
        "using: num_layers * 2 * num_kv_heads * head_dim * dtype_bytes.",
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
    parser.add_argument(
        "--event-plane",
        type=str,
        choices=["nats", "zmq"],
        default=os.environ.get("DYN_EVENT_PLANE", "nats"),
        help="Determines how events are published [nats|zmq]",
    )

    args = parser.parse_args(argv)
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

    # Parse and validate zmq_kv_events_ports (same comma-separated format as bootstrap_ports)
    args.zmq_kv_events_ports_list = parse_bootstrap_ports(args.zmq_kv_events_ports)
    if args.zmq_kv_events_ports_list:
        if len(args.zmq_kv_events_ports_list) != args.num_workers:
            raise ValueError(
                f"--zmq-kv-events-ports must have exactly --num-workers ({args.num_workers}) ports, "
                f"got {len(args.zmq_kv_events_ports_list)}: {args.zmq_kv_events_ports_list}"
            )

    # Parse and validate zmq_replay_ports
    args.zmq_replay_ports_list = parse_bootstrap_ports(args.zmq_replay_ports)
    if args.zmq_replay_ports_list:
        if not args.zmq_kv_events_ports_list:
            raise ValueError("--zmq-replay-ports requires --zmq-kv-events-ports")
        if len(args.zmq_replay_ports_list) != args.num_workers:
            raise ValueError(
                f"--zmq-replay-ports must have exactly --num-workers ({args.num_workers}) ports, "
                f"got {len(args.zmq_replay_ports_list)}: {args.zmq_replay_ports_list}"
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


def _default_kv_transfer_bandwidth_gbps() -> float:
    from .utils.kv_cache import DEFAULT_KV_TRANSFER_BANDWIDTH_GBPS

    return DEFAULT_KV_TRANSFER_BANDWIDTH_GBPS

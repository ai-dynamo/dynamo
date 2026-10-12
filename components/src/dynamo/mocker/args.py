#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import tempfile
from pathlib import Path

from dynamo.common.configuration.groups.router_args import (
    WorkerRouterConfig,
    add_worker_router_arguments,
)
from dynamo.common.configuration.utils import Deprecated
from dynamo.common.utils.namespace import get_worker_namespace

from . import __version__

DYN_NAMESPACE = get_worker_namespace()
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"
DEFAULT_PREFILL_ENDPOINT = f"dyn://{DYN_NAMESPACE}.prefill.generate"

logger = logging.getLogger(__name__)
_SGLANG_ALIAS_REMOVAL = "Dynamo 1.8.0 (two releases after 1.6.0)"


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
    from dynamo.mocker.config import normalize_mocker_config

    engine_defaults = normalize_mocker_config()["engine"]
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

    # Engine CLI options (lowered to the AISimulate configuration)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        dest="num_gpu_blocks",
        default=None,
        help="Explicit usable GPU-block capacity per data-parallel rank for the mock "
        "KV cache. When unset, AIS-backed mocker estimates the value; non-AIS "
        "mocker uses 16384.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Token block size for KV cache blocks. When unset, the default "
        "depends on engine: vLLM 64, SGLang 1, TRTLLM 32.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length, including prompt and generated tokens. "
        "When omitted, no model-length limit is enforced.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=engine_defaults["max_num_seqs"],
        help="Maximum number of sequences per iteration (default: 256)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=engine_defaults["max_num_batched_tokens"],
        help="Maximum number of batched tokens per iteration (default: 8192)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        dest="enable_prefix_caching",
        default=engine_defaults["enable_prefix_caching"],
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
        default=engine_defaults["enable_chunked_prefill"],
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
        default=engine_defaults["preemption_mode"],
        help="Preemption mode for decode eviction under memory pressure. "
        "'lifo' (default) evicts the newest request (matches vLLM v1), "
        "'fifo' evicts the oldest request.",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=engine_defaults["speedup_ratio"],
        help="Speedup ratio for mock execution (default: 1.0). Use 0 for infinite speedup (no simulation delays).",
    )
    parser.add_argument(
        "--decode-speedup-ratio",
        type=float,
        default=engine_defaults["decode_speedup_ratio"],
        help="Additional speedup multiplier applied only to decode steps (default: 1.0). "
        "Models speculative decoding (e.g. Eagle) where decode throughput improves "
        "without affecting prefill latency. Effective decode speedup is speedup_ratio * decode_speedup_ratio.",
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
        "--ais-perf-model",
        "--aic-perf-model",
        dest="ais_perf_model",
        action="store_true",
        default=False,
        help="Use AISimulate's perf model directly for latency prediction. "
        "Requires aisimulate installed.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="GPU memory fraction for AIS KV capacity estimation with vLLM "
        "(default: 0.9).",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Static memory fraction for AIS KV capacity estimation with SGLang "
        "(default: 0.88).",
    )
    parser.add_argument(
        "--free-gpu-memory-fraction",
        type=float,
        default=None,
        help="Fraction of free GPU memory (after model load) for the KV cache, "
        "for AIS KV capacity estimation with TRT-LLM (default: 0.9).",
    )
    parser.add_argument(
        "--ais-system",
        "--aic-system",
        dest="ais_system",
        type=str,
        default=None,
        help="AIS system name (e.g., 'h200_sxm'). Used with --ais-perf-model.",
    )
    parser.add_argument(
        "--ais-backend",
        "--aic-backend",
        dest="ais_backend",
        type=str,
        default=None,
        help="AIS backend name used for perf database lookups. When unset, "
        "falls back to --engine-type. Set this to decouple the AIS perf model "
        "from the simulated engine type (e.g. simulate with vllm while using "
        "trtllm AIS data).",
    )
    parser.add_argument(
        "--ais-backend-version",
        "--aic-backend-version",
        dest="ais_backend_version",
        type=str,
        default=None,
        help="AIS performance-database version: 'current', 'previous', or 'next' "
        "when available, or a version assigned to one of those slots. "
        "Defaults to the release database's 'current' slot.",
    )
    parser.add_argument(
        "--ais-tp-size",
        "--aic-tp-size",
        dest="ais_tp_size",
        type=int,
        default=None,
        help="Tensor parallel size for AIS latency prediction (default: 1). "
        "Only affects AIS performance model lookups, not mocker scheduling.",
    )
    parser.add_argument(
        "--ais-moe-tp-size",
        "--aic-moe-tp-size",
        dest="ais_moe_tp_size",
        type=int,
        default=None,
        help="MoE tensor-parallel size for AIS latency prediction. "
        "Required for MoE models. Constraint: ais_tp_size * ais_attention_dp_size == ais_moe_tp_size * ais_moe_ep_size.",
    )
    parser.add_argument(
        "--ais-moe-ep-size",
        "--aic-moe-ep-size",
        dest="ais_moe_ep_size",
        type=int,
        default=None,
        help="MoE expert-parallel size for AIS latency prediction. "
        "Required for MoE models. Constraint: ais_tp_size * ais_attention_dp_size == ais_moe_tp_size * ais_moe_ep_size.",
    )
    parser.add_argument(
        "--ais-attention-dp-size",
        "--aic-attention-dp-size",
        dest="ais_attention_dp_size",
        type=int,
        default=None,
        help="Attention data-parallel size for AIS latency prediction (default: 1). "
        "Corresponds to the 'dp' dimension in AIS CLI output.",
    )
    parser.add_argument(
        "--ais-nextn",
        "--aic-nextn",
        dest="ais_nextn",
        type=int,
        default=None,
        help="[EXPERIMENTAL] Number of MTP draft tokens to sample (1-5).",
    )
    parser.add_argument(
        "--ais-nextn-accept-rates",
        "--aic-nextn-accept-rates",
        dest="ais_nextn_accept_rates",
        type=str,
        default=None,
        help=(
            "[EXPERIMENTAL] Comma-separated conditional MTP acceptance rates. "
            "Entry i is P(draft i accepted | all earlier drafts were accepted)."
        ),
    )
    parser.add_argument(
        "--ais-mtp-seed",
        "--aic-mtp-seed",
        dest="ais_mtp_seed",
        type=int,
        default=engine_defaults["aic_mtp_seed"],
        help="[EXPERIMENTAL] Base RNG seed for mocker MTP burst sampling.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of mocker workers to launch in the same process (default: 1). "
        "All workers share the same tokio runtime and thread pool.",
    )

    from dynamo.common.configuration.groups.ais_perf_args import parse_ais_perf_config

    parser.add_argument(
        "--ais-perf-config",
        type=parse_ais_perf_config,
        default=None,
        help="Complete ForwardPassPerfModelConfig as JSON or a JSON/YAML file.",
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
    parser.add_argument(
        "--response-replay-trace-path",
        type=str,
        default=None,
        help=(
            "Optional Mooncake JSONL trace containing output_token_ids for "
            "output_replay_id annotation lookup."
        ),
    )

    # Engine type selection
    parser.add_argument(
        "--engine-type",
        type=str,
        default=engine_defaults["backend"],
        help="Engine simulation type: 'vllm' (default), 'sglang', or 'trtllm'.",
    )

    # SGLang-specific configuration
    parser.add_argument(
        "--sglang-schedule-policy",
        type=str,
        default=None,
        help="SGLang scheduling policy: 'fifo' (default) or 'lpm' (longest prefix match). "
        "The 'fcfs' alias is deprecated and will be removed in Dynamo 1.8.0.",
    )
    parser.add_argument(
        "--sglang-page-size",
        type=int,
        default=None,
        help="Deprecated SGLang alias for --block-size; removed in Dynamo 1.8.0. "
        "Ignored for other backends.",
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
    parser.add_argument(
        "--sglang-generate",
        action="store_true",
        default=False,
        help="Serve native streaming SGLang /generate requests (default: disabled).",
    )

    # TensorRT-LLM-specific configuration
    parser.add_argument(
        "--trtllm-capacity-scheduler-policy",
        type=str,
        default=None,
        help="TRT-LLM capacity scheduler policy. v1 supports only "
        "'guaranteed_no_evict' (default).",
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
        default=engine_defaults["kv_transfer_bandwidth"],
        help="KV cache transfer bandwidth in GB/s for disaggregated serving latency simulation. "
        "When unset, uses the AISimulate engine default. Set to 0 to disable KV transfer delay. "
        "For intra-node NVLink, typical value is ~450.",
    )
    parser.add_argument(
        "--kv-transfer-timing-mode",
        default=engine_defaults["kv_transfer_timing_mode"],
        help="Physical KV footprint used for coordinated disaggregated transfer timing.",
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
        "--num-host-blocks",
        type=int,
        default=None,
        help="Enable native vLLM G2 (host) KV offload with this per-DP-rank host cache "
        "capacity in blocks. Host blocks are sized by --kv-bytes-per-token. Requires "
        "prefix caching.",
    )
    for direction in ("d2h", "h2d"):
        parser.add_argument(
            f"--host-offload-{direction}-bandwidth-gbps",
            type=float,
            default=None,
            help=f"Per-DP-rank native G2 {direction.upper()} bandwidth in decimal GB/s "
            "(AISimulate default when unset; 0 is unlimited). Requires --num-host-blocks.",
        )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=0.0,
        help=(
            "Delay in seconds between launching each worker. "
            "Set to 0 to disable staggering (default). "
            "Use -1 for auto mode (0.1s for 33-128 workers, 0.2s for >128 workers, 0 otherwise)."
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
        choices=["nats", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|tcp]",
    )
    parser.add_argument(
        "--response-plane",
        type=str,
        choices=["tcp", "quic"],
        default=os.environ.get("DYN_RESPONSE_PLANE", "tcp"),
        help="Select the response transport. Frontend and workers must match.",
    )
    parser.add_argument(
        "--event-plane",
        type=str,
        choices=["nats", "zmq"],
        default=os.environ.get("DYN_EVENT_PLANE"),
        help="Determines how events are published [nats|zmq]. If unset, "
        "auto-detected from --discovery-backend (zmq for file/mem, nats "
        "for etcd/kubernetes).",
    )

    # Same flags the frontend and engine backends expose, so a mocker can stand
    # in for a real worker set when exercising per-role routing.
    add_worker_router_arguments(parser)

    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    spellings: dict[str, str] = {}
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag.startswith(("--ais-", "--aic-")):
            canonical = flag.replace("--aic-", "--ais-", 1)
            if canonical in spellings and spellings[canonical] != flag:
                parser.error(
                    f"{canonical} and its legacy --aic spelling cannot be combined"
                )
            spellings[canonical] = flag
    args = parser.parse_args(argv)
    if args.sglang_schedule_policy == "fcfs":
        Deprecated(
            "--sglang-schedule-policy fifo", remove_in=_SGLANG_ALIAS_REMOVAL
        ).warn("--sglang-schedule-policy fcfs")
        args.sglang_schedule_policy = "fifo"
    if args.sglang_page_size is not None:
        Deprecated("--block-size", remove_in=_SGLANG_ALIAS_REMOVAL).warn(
            "--sglang-page-size"
        )
        if args.engine_type == "sglang":
            if args.block_size is not None and args.block_size != args.sglang_page_size:
                parser.error("--sglang-page-size and --block-size must match")
            args.block_size = args.sglang_page_size
    # Collect them into their own config object, matching the backends.
    args.router_advertisement = WorkerRouterConfig.from_cli_args(args)

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

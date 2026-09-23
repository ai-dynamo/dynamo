#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across router benchmark scripts."""

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

# Default values
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_SEED = 0
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MOONCAKE_BLOCK_SIZE = 512


def setup_logger(name: str) -> logging.Logger:
    """Setup and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def add_common_args(parser):
    """Add common CLI arguments shared across benchmark scripts."""
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--use-expected-osl",
        action="store_true",
        help="Pass agent_hints.osl through extra.nvext for router output block tracking",
    )
    parser.add_argument(
        "--verify-worker-participation",
        action="store_true",
        help="Capture worker IDs from measured responses and write a participation report",
    )
    parser.add_argument(
        "--minimum-prefill-workers",
        type=_non_negative_int,
        default=0,
        help="Fail unless at least this many prefill workers are observed (default: 0)",
    )
    parser.add_argument(
        "--minimum-decode-workers",
        type=_non_negative_int,
        default=0,
        help="Fail unless at least this many decode workers are observed (default: 0)",
    )


def add_synthesis_args(parser):
    """Add CLI arguments for trace dataset synthesis, shared across benchmark scripts."""
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data_benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="mooncake_trace.jsonl",
        help="Path to the input mooncake-style trace dataset file",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Number of requests to synthesize (default: use all from input file)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=1.0,
        help="Factor to speed up request intervals (default: 1.0)",
    )
    parser.add_argument(
        "--prefix-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for prefix lengths (default: 1.0)",
    )
    parser.add_argument(
        "--prefix-root-multiplier",
        type=int,
        default=1,
        help="Number of times to replicate the core radix tree (default: 1)",
    )
    parser.add_argument(
        "--prompt-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for leaf path lengths (default: 1.0, use <1 for shorter prompts)",
    )
    parser.add_argument(
        "--max-isl",
        type=int,
        default=None,
        help="Maximum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--min-isl",
        type=int,
        default=None,
        help="Minimum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--max-rejections",
        type=int,
        default=10000,
        help="Maximum consecutive ISL rejections before failing (default: 10000)",
    )
    parser.add_argument(
        "--min-osl",
        type=int,
        default=None,
        help="Minimum output sequence length - clips values below this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--max-osl",
        type=int,
        default=None,
        help="Maximum output sequence length - clips values above this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--osl-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for output sequence lengths (default: 1.0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_MOONCAKE_BLOCK_SIZE,
        help=f"Block size for prefilling and decoding (default: {DEFAULT_MOONCAKE_BLOCK_SIZE})",
    )


def resolve_tokenizer(args):
    """Set tokenizer to model if not specified."""
    if args.tokenizer is None:
        args.tokenizer = args.model


def _non_negative_int(value: str) -> int:
    """Reject negative minima before they make threshold checks vacuously pass."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def worker_participation_requested(args: Any) -> bool:
    """Treat a positive worker minimum as implicit participation opt-in."""
    return (
        args.verify_worker_participation
        or args.minimum_prefill_workers > 0
        or args.minimum_decode_workers > 0
    )


def get_common_aiperf_flags(
    capture_worker_participation: bool = False,
    nvext: dict[str, Any] | None = None,
) -> list[str]:
    """Return common aiperf flags used across benchmarks."""
    flags = [
        "--endpoint-type",
        "chat",
        "--streaming",
        "--extra-inputs",
        "ignore_eos:true",
        "--no-gpu-telemetry",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]

    # AIPerf converts repeated top-level extra-input keys to a dict, so emit
    # nvext once to preserve caller-provided router hints.
    nvext = copy.deepcopy(nvext) if nvext else {}
    if capture_worker_participation:
        extra_fields = nvext.setdefault("extra_fields", [])
        if "worker_id" not in extra_fields:
            extra_fields.append("worker_id")
        flags.extend(["--export-level", "raw"])
    if nvext:
        flags.extend(["--extra-inputs", json.dumps({"nvext": nvext})])
    return flags


def get_aiperf_cmd_for_trace(
    model,
    tokenizer,
    input_dataset,
    artifact_dir,
    seed,
    block_size,
    url="http://localhost:8888",
    capture_worker_participation=False,
):
    """Build the aiperf CLI command for a mooncake trace run."""
    cmd = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--url",
        url,
        "--input-file",
        f"{input_dataset}",
        "--custom-dataset-type",
        "mooncake_trace",
        "--fixed-schedule",
        "--fixed-schedule-auto-offset",
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
    ]
    cmd.extend(get_common_aiperf_flags(capture_worker_participation))
    return cmd


def _response_payloads(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode JSON payloads from one raw AIPerf text or SSE response."""
    encoded_payloads = []
    text = response.get("text")
    if isinstance(text, str):
        encoded_payloads.append(text)

    packets = response.get("packets")
    if isinstance(packets, list):
        encoded_payloads.extend(
            packet["value"]
            for packet in packets
            if isinstance(packet, dict)
            and packet.get("name") == "data"
            and isinstance(packet.get("value"), str)
        )

    payloads = []
    for encoded_payload in encoded_payloads:
        if encoded_payload == "[DONE]":
            continue
        try:
            payload = json.loads(encoded_payload)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _add_worker_id(worker_ids: set[int | str], value: Any) -> bool:
    """Reject bool because JSON booleans deserialize as Python int subclasses."""
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        worker_ids.add(value)
        return True
    return False


def collect_worker_participation(artifact_root: str | Path) -> dict[str, Any]:
    """Collect worker IDs from profiling requests in AIPerf raw exports."""
    root = Path(artifact_root)
    raw_exports = sorted(root.rglob("profile_export_raw.jsonl"))
    if not raw_exports:
        raise FileNotFoundError(
            f"No profile_export_raw.jsonl found under {artifact_root}"
        )

    profiling_requests = 0
    requests_with_worker_id = 0
    prefill_worker_ids: set[int | str] = set()
    decode_worker_ids: set[int | str] = set()

    for raw_export in raw_exports:
        with raw_export.open() as raw_records:
            for line_number, line in enumerate(raw_records, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {raw_export}:{line_number}"
                    ) from error

                metadata = record.get("metadata") or {}
                if metadata.get("benchmark_phase") not in (None, "profiling"):
                    continue

                profiling_requests += 1
                request_has_worker_id = False
                for response in record.get("responses") or []:
                    if not isinstance(response, dict):
                        continue
                    for payload in _response_payloads(response):
                        nvext = payload.get("nvext")
                        if not isinstance(nvext, dict):
                            continue
                        worker_id = nvext.get("worker_id")
                        if not isinstance(worker_id, dict):
                            continue

                        found_prefill = _add_worker_id(
                            prefill_worker_ids,
                            worker_id.get("prefill_worker_id"),
                        )
                        found_decode = _add_worker_id(
                            decode_worker_ids,
                            worker_id.get("decode_worker_id"),
                        )
                        request_has_worker_id |= found_prefill or found_decode

                requests_with_worker_id += int(request_has_worker_id)

    return {
        "profiling_requests": profiling_requests,
        "requests_with_worker_id": requests_with_worker_id,
        "observed_prefill_worker_ids": sorted(prefill_worker_ids, key=str),
        "observed_decode_worker_ids": sorted(decode_worker_ids, key=str),
        "raw_export_files": [str(path.relative_to(root)) for path in raw_exports],
    }


def validate_worker_participation(
    artifact_root: str | Path,
    minimum_prefill_workers: int,
    minimum_decode_workers: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Write and validate worker participation from measured AIPerf responses."""
    report = collect_worker_participation(artifact_root)
    report["minimum_prefill_workers"] = minimum_prefill_workers
    report["minimum_decode_workers"] = minimum_decode_workers

    report_path = Path(artifact_root) / "worker_participation.json"
    with report_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    prefill_count = len(report["observed_prefill_worker_ids"])
    decode_count = len(report["observed_decode_worker_ids"])
    logger.info(
        "Worker participation: %d/%d profiling requests exposed worker IDs; "
        "prefill=%d, decode=%d",
        report["requests_with_worker_id"],
        report["profiling_requests"],
        prefill_count,
        decode_count,
    )
    logger.info("Worker participation report: %s", report_path)

    failures = []
    if prefill_count == 0 and decode_count == 0:
        failures.append("no worker IDs were returned")
    if prefill_count < minimum_prefill_workers:
        failures.append(
            f"observed {prefill_count} prefill workers, required {minimum_prefill_workers}"
        )
    if decode_count < minimum_decode_workers:
        failures.append(
            f"observed {decode_count} decode workers, required {minimum_decode_workers}"
        )
    if failures:
        raise RuntimeError(
            "Worker participation validation failed: " + "; ".join(failures)
        )

    return report


def add_worker_id_extra_field(request):
    """Request worker attribution without replacing per-turn Dynamo fields."""
    extra = request.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        request["extra"] = extra

    nvext = extra.get("nvext")
    if not isinstance(nvext, dict):
        nvext = {}
        extra["nvext"] = nvext

    extra_fields = nvext.get("extra_fields")
    if extra_fields is None:
        extra_fields = []
        nvext["extra_fields"] = extra_fields
    elif not isinstance(extra_fields, list):
        raise ValueError("extra.nvext.extra_fields must be a list")

    if "worker_id" not in extra_fields:
        extra_fields.append("worker_id")


def set_trace_agent_hint(
    request,
    name,
    value,
    capture_worker_participation=False,
):
    """Set an agent hint in the trace envelope forwarded by AIPerf."""
    extra = request.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        request["extra"] = extra

    nvext = extra.get("nvext")
    if not isinstance(nvext, dict):
        nvext = {}
        extra["nvext"] = nvext

    agent_hints = nvext.get("agent_hints")
    if not isinstance(agent_hints, dict):
        agent_hints = {}
        nvext["agent_hints"] = agent_hints

    agent_hints[name] = value
    if capture_worker_participation:
        add_worker_id_extra_field(request)


def add_expected_osl(request, capture_worker_participation=False):
    """Add the trace output length as the router's expected OSL hint."""
    osl = request.get("output_length", request.get("output_tokens", 0))
    set_trace_agent_hint(request, "osl", osl, capture_worker_participation)


def tag_requests_with_priority(
    requests,
    priority,
    capture_worker_participation=False,
):
    """Return request copies with extra.nvext.agent_hints.priority merged in."""
    tagged_requests = []
    for request in requests:
        tagged_request = copy.deepcopy(request)
        set_trace_agent_hint(
            tagged_request,
            "priority",
            priority,
            capture_worker_participation,
        )
        tagged_requests.append(tagged_request)
    return tagged_requests


def prepare_trace_dataset(args, output_dir, logger):
    """Prepare a trace dataset, optionally synthesizing or modifying it.

    Handles three paths:
    1. No synthesis needed: use the original dataset as-is
    2. Metadata injection only: add expected OSL and/or worker attribution
    3. Full synthesis: generate synthetic data from the input dataset

    Returns:
        tuple[list[dict], str]: (list of request dicts, path to the trace file)
    """
    needs_synthesis = (
        args.num_requests is not None
        or args.speedup_ratio != 1.0
        or args.prefix_len_multiplier != 1.0
        or args.prefix_root_multiplier != 1
        or args.prompt_len_multiplier != 1.0
        or args.osl_multiplier != 1.0
        or args.max_isl is not None
        or args.min_isl is not None
        or args.min_osl is not None
        or args.max_osl is not None
    )

    capture_worker_participation = worker_participation_requested(args)

    if (
        not needs_synthesis
        and not args.use_expected_osl
        and not capture_worker_participation
    ):
        # No synthesis or modification needed, use original dataset
        trace_dataset_path = args.input_dataset
        logger.info(
            f"Using original trace dataset (no synthesis parameters modified): {trace_dataset_path}"
        )
        requests = []
        with open(args.input_dataset, "r") as f:
            for line in f:
                requests.append(json.loads(line.strip()))
        return requests, trace_dataset_path

    if not needs_synthesis:
        # Apply request metadata without changing the trace's timing or shape.
        if args.use_expected_osl:
            logger.info("Injecting agent_hints.osl into original trace dataset...")
        if capture_worker_participation:
            logger.info("Requesting worker attribution in original trace dataset...")

        requests = []
        with open(args.input_dataset, "r") as f:
            for line in f:
                requests.append(json.loads(line.strip()))

        for request in requests:
            if args.use_expected_osl:
                add_expected_osl(request, capture_worker_participation)
            elif capture_worker_participation:
                add_worker_id_extra_field(request)

        trace_name = (
            "trace_with_expected_osl.jsonl"
            if args.use_expected_osl
            else "trace_with_worker_participation.jsonl"
        )
        trace_dataset_path = os.path.join(output_dir, trace_name)
        with open(trace_dataset_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Modified trace data saved to: {trace_dataset_path}")
        return requests, trace_dataset_path

    # Generate synthetic data based on input dataset
    logger.info("Generating synthetic trace data...")
    logger.info(f"  Base dataset: {args.input_dataset}")
    logger.info(f"  Num requests: {args.num_requests if args.num_requests else 'all'}")
    logger.info(f"  Speedup ratio: {args.speedup_ratio}")
    logger.info(f"  Prefix len multiplier: {args.prefix_len_multiplier}")
    logger.info(f"  Prefix root multiplier: {args.prefix_root_multiplier}")
    logger.info(f"  Prompt len multiplier: {args.prompt_len_multiplier}")
    logger.info(f"  OSL multiplier: {args.osl_multiplier}")
    logger.info(
        f"  Max ISL: {args.max_isl if args.max_isl else 'no limit'} (filtering)"
    )
    logger.info(
        f"  Min ISL: {args.min_isl if args.min_isl else 'no limit'} (filtering)"
    )
    logger.info(
        f"  Min OSL: {args.min_osl if args.min_osl else 'no clipping'} (clipping)"
    )
    logger.info(
        f"  Max OSL: {args.max_osl if args.max_osl else 'no clipping'} (clipping)"
    )
    logger.info(f"  Random seed: {args.seed}")

    # Synthetic generation has optional dependencies absent from slim CPU test images.
    # Keep the request helpers and trace-file path usable without those dependencies.
    from prefix_data_generator.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        args.input_dataset,
        block_size=args.block_size,
        speedup_ratio=args.speedup_ratio,
        prefix_len_multiplier=args.prefix_len_multiplier,
        prefix_root_multiplier=args.prefix_root_multiplier,
        prompt_len_multiplier=args.prompt_len_multiplier,
        osl_multiplier=args.osl_multiplier,
        seed=args.seed,
    )

    if args.num_requests is None:
        with open(args.input_dataset, "r") as f:
            num_requests = sum(1 for _ in f)
        logger.info(f"Using all {num_requests} requests from input dataset")
    else:
        num_requests = args.num_requests

    requests = synthesizer.synthesize_requests(
        num_requests,
        max_isl=args.max_isl,
        min_isl=args.min_isl,
        max_rejections=args.max_rejections,
        min_osl=args.min_osl,
        max_osl=args.max_osl,
    )
    logger.info(f"Generated {len(requests)} synthetic requests")

    trace_dataset_path = os.path.join(output_dir, "synthetic_trace.jsonl")

    if args.use_expected_osl or capture_worker_participation:
        for request in requests:
            if args.use_expected_osl:
                add_expected_osl(request, capture_worker_participation)
            else:
                add_worker_id_extra_field(request)

    if args.use_expected_osl:
        logger.info("Injected agent_hints.osl into extra.nvext for each request")
    if capture_worker_participation:
        logger.info("Requested worker attribution in extra.nvext for each request")

    with open(trace_dataset_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    logger.info(f"Synthetic trace data saved to: {trace_dataset_path}")
    return requests, trace_dataset_path

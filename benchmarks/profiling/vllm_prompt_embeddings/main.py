# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile sequential synchronous vLLM generation from prompt embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from vllm import LLM, SamplingParams
except (ImportError, AttributeError):  # vLLM is optional for CPU-only tests.
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one synchronous prompt-embedding experiment."""

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    requests: int = 100
    warmup_requests: int = 20
    prompt_tokens: int = 515
    output_tokens: int = 75
    seed: int = 0
    block_size: int = 16
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.90

    @property
    def prefix_cache_remainder_tokens(self) -> int:
        remainder = self.prompt_tokens % self.block_size
        return remainder or self.block_size

    @property
    def cudagraph_capture_sizes(self) -> list[int]:
        return sorted({1, self.prefix_cache_remainder_tokens, self.prompt_tokens})

    def validate(self) -> None:
        if self.requests <= 0:
            raise ValueError("requests must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests cannot be negative")
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.prompt_tokens + self.output_tokens > self.max_model_len:
            raise ValueError("prompt and output tokens exceed max_model_len")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")


def enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def tensor_sha256(tensor: torch.Tensor) -> str:
    byte_view = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def create_prompt_embeddings(
    *,
    prompt_tokens: int,
    hidden_dimension: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Create one deterministic CPU tensor that every request reuses."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    embeddings = torch.randn(
        (prompt_tokens, hidden_dimension),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return embeddings.mul_(0.02).to(dtype=dtype).contiguous()


def create_sampling_params(
    output_tokens: int,
    seed: int,
    sampling_params_factory: Callable[..., Any],
) -> Any:
    return sampling_params_factory(
        temperature=0.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        seed=seed,
        detokenize=False,
    )


def validate_generation(outputs: Any, expected_output_tokens: int) -> dict[str, Any]:
    output_count = len(outputs) if isinstance(outputs, list) else 0
    if output_count != 1:
        raise ValueError(f"expected one RequestOutput, received {output_count}")
    candidates = getattr(outputs[0], "outputs", None)
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    if candidate_count != 1:
        raise ValueError(f"expected one completion, received {candidate_count}")
    completion = candidates[0]
    token_ids = list(getattr(completion, "token_ids", []))
    if len(token_ids) != expected_output_tokens:
        raise ValueError(
            f"OSL mismatch: received {len(token_ids)}, "
            f"expected {expected_output_tokens}"
        )
    return {
        "request_id": str(getattr(outputs[0], "request_id", "")),
        "output_tokens": len(token_ids),
        "finish_reason": getattr(completion, "finish_reason", None),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def execute_requests(
    llm: Any,
    prompt_embeddings: torch.Tensor,
    sampling_params: Any,
    *,
    count: int,
    expected_output_tokens: int,
    phase: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for request_index in range(count):
        started_ns = time.perf_counter_ns()
        outputs = llm.generate(
            [{"prompt_embeds": prompt_embeddings}],
            sampling_params,
            use_tqdm=False,
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        record = validate_generation(outputs, expected_output_tokens)
        record.update(
            {
                "phase": phase,
                "request_index": request_index,
                "latency_ms": elapsed_ns / 1_000_000,
            }
        )
        records.append(record)
    return records


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def resolved_engine_config(llm: Any) -> dict[str, Any]:
    vllm_config = llm.llm_engine.vllm_config
    compilation = vllm_config.compilation_config
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    return {
        "cuda_graph_mode": enum_name(compilation.cudagraph_mode),
        "cuda_graph_capture_sizes": list(compilation.cudagraph_capture_sizes),
        "enforce_eager": bool(model.enforce_eager),
        "prefix_caching": bool(cache.enable_prefix_caching),
        "block_size": int(cache.block_size),
        "model_revision": getattr(model.hf_config, "_commit_hash", None),
    }


def validate_resolved_engine(
    resolved: dict[str, Any], config: ExperimentConfig
) -> None:
    failures: list[str] = []
    if resolved["cuda_graph_mode"] != "FULL":
        failures.append(
            f"cuda_graph_mode={resolved['cuda_graph_mode']!r}, expected 'FULL'"
        )
    missing_sizes = sorted(
        set(config.cudagraph_capture_sizes) - set(resolved["cuda_graph_capture_sizes"])
    )
    if missing_sizes:
        failures.append(f"CUDA graph capture sizes are missing {missing_sizes}")
    if resolved["enforce_eager"]:
        failures.append("enforce_eager unexpectedly resolved to true")
    if not resolved["prefix_caching"]:
        failures.append("prefix caching unexpectedly resolved to false")
    if resolved["block_size"] != config.block_size:
        failures.append(
            f"block_size={resolved['block_size']}, expected {config.block_size}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))


def write_results(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requests.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    llm_factory: Callable[..., Any],
    sampling_params_factory: Callable[..., Any],
) -> dict[str, Any]:
    """Run warmups, profile measured requests, and write validated results."""
    config.validate()
    llm = llm_factory(
        model=config.model,
        tensor_parallel_size=1,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        block_size=config.block_size,
        enable_prompt_embeds=True,
        enable_prefix_caching=True,
        generation_config="vllm",
        enforce_eager=False,
        seed=config.seed,
        profiler_config={"profiler": "cuda"},
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": config.cudagraph_capture_sizes,
        },
    )
    model_config = llm.model_config
    hidden_dimension = int(model_config.get_hidden_size())
    inputs_embeds_size = int(model_config.get_inputs_embeds_size())
    if inputs_embeds_size != hidden_dimension:
        raise RuntimeError(
            "the model input embedding width does not equal its hidden dimension: "
            f"{inputs_embeds_size} != {hidden_dimension}"
        )
    prompt_embeddings = create_prompt_embeddings(
        prompt_tokens=config.prompt_tokens,
        hidden_dimension=hidden_dimension,
        dtype=model_config.dtype,
        seed=config.seed,
    )
    if prompt_embeddings.shape != (config.prompt_tokens, hidden_dimension):
        raise RuntimeError(f"unexpected embedding shape {prompt_embeddings.shape}")
    if prompt_embeddings.device.type != "cpu" or not prompt_embeddings.is_contiguous():
        raise RuntimeError("prompt embeddings must be contiguous CPU tensors")

    resolved = resolved_engine_config(llm)
    validate_resolved_engine(resolved, config)
    sampling_params = create_sampling_params(
        config.output_tokens,
        config.seed,
        sampling_params_factory,
    )

    execute_requests(
        llm,
        prompt_embeddings,
        sampling_params,
        count=config.warmup_requests,
        expected_output_tokens=config.output_tokens,
        phase="warmup",
    )

    profile_started = False
    try:
        llm.start_profile()
        profile_started = True
        measured_started_ns = time.perf_counter_ns()
        records = execute_requests(
            llm,
            prompt_embeddings,
            sampling_params,
            count=config.requests,
            expected_output_tokens=config.output_tokens,
            phase="measured",
        )
        measured_elapsed_s = (time.perf_counter_ns() - measured_started_ns) / 1e9
    finally:
        if profile_started:
            llm.stop_profile()

    latencies = [float(record["latency_ms"]) for record in records]
    summary = {
        "accepted": len(records) == config.requests
        and all(record["output_tokens"] == config.output_tokens for record in records),
        "config": asdict(config),
        "embedding": {
            "shape": list(prompt_embeddings.shape),
            "dtype": str(prompt_embeddings.dtype),
            "device": prompt_embeddings.device.type,
            "contiguous": prompt_embeddings.is_contiguous(),
            "sha256": tensor_sha256(prompt_embeddings),
        },
        "resolved_engine": resolved,
        "sampling": {
            "temperature": 0.0,
            "min_tokens": config.output_tokens,
            "max_tokens": config.output_tokens,
            "ignore_eos": True,
            "seed": config.seed,
            "detokenize": False,
        },
        "results": {
            "requests": len(records),
            "total_seconds": measured_elapsed_s,
            "request_throughput": len(records) / measured_elapsed_s,
            "latency_ms": {
                "avg": sum(latencies) / len(latencies),
                "p50": percentile(latencies, 50),
                "p90": percentile(latencies, 90),
                "p99": percentile(latencies, 99),
            },
        },
    }
    write_results(output_dir, records, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile synchronous vLLM generation from prompt embeddings."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=515)
    parser.add_argument("--output-tokens", type=int, default=75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if LLM is None or SamplingParams is None:
        raise SystemExit("vLLM is required to run this experiment")
    config = ExperimentConfig(
        model=args.model,
        requests=args.requests,
        warmup_requests=args.warmup_requests,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        seed=args.seed,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    summary = run_experiment(
        config,
        args.output_dir,
        llm_factory=LLM,
        sampling_params_factory=SamplingParams,
    )
    results = summary["results"]
    latency = results["latency_ms"]
    print(
        "EXPERIMENT_COMPLETE "
        f"requests={results['requests']} "
        f"osl={config.output_tokens} "
        f"avg_ms={latency['avg']:.3f} "
        f"throughput={results['request_throughput']:.3f}"
    )


if __name__ == "__main__":
    main()

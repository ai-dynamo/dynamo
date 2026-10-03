"""Write CR/config/engine.json: the campaign's single engine source of truth."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from aisimulate_core.sdk import ForwardPassPerfModelConfig

from dynamo._internal.ais import estimate_canonical_num_gpu_blocks

CR = Path("<campaign-root>")

# Portable AIS identity: no resolved systems_paths, so remote bundles re-normalize locally.
AIS_INPUT = {
    "model": "Qwen/Qwen3-32B",
    "system": "h100_sxm",
    "backend": "vllm",
    "backend_version": "0.24.0",
    "worker_type": "aggregated",
    "tp": 2,
}
BLOCK_SIZE = 16
MAX_NUM_SEQS = 1024
MAX_NUM_BATCHED_TOKENS = 8192
MAX_MODEL_LEN = 131072


def main() -> None:
    canonical = ForwardPassPerfModelConfig(**AIS_INPUT).to_dict()
    num_gpu_blocks = estimate_canonical_num_gpu_blocks(
        canonical,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
    )
    mock_engine_args = {
        "engine_type": "vllm",
        "worker_type": "aggregated",
        "block_size": BLOCK_SIZE,
        "num_gpu_blocks": num_gpu_blocks,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "speedup_ratio": 1.0,
        "decode_speedup_ratio": 1.0,
        "dp_size": 1,
        "ais_perf_config": AIS_INPUT,
    }
    engine = {
        "schema": "learned-routing.engine.v1",
        "model": AIS_INPUT["model"],
        "hardware": AIS_INPUT["system"],
        "backend": AIS_INPUT["backend"],
        "backend_version": AIS_INPUT["backend_version"],
        "tp": AIS_INPUT["tp"],
        "mode": "aggregated",
        "max_model_len": MAX_MODEL_LEN,
        "ais_perf_config": AIS_INPUT,
        "ais_perf_config_canonical_local": canonical,
        "mock_engine_args": mock_engine_args,
        "kv_capacity": {
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": BLOCK_SIZE,
            "tokens_per_worker": num_gpu_blocks * BLOCK_SIZE,
            "source": "dynamo._internal.ais.estimate_canonical_num_gpu_blocks"
            f"(block_size={BLOCK_SIZE}, max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS},"
            f" max_num_seqs={MAX_NUM_SEQS}); pinned so every replay uses identical capacity",
        },
        "replay_call": {
            "engine_args": "MockEngineArgs.from_json(json.dumps(engine['mock_engine_args']))",
            "top_level_ais_perf_config": None,
            "router_prefill_load_model": "none",
            "note": "Top-level run_trace_replay(ais_perf_config=...) is the router's AIS prefill-load"
            " estimator; it must stay None (no AIS coupling in routing). Engine timing comes only"
            " from mock_engine_args.ais_perf_config.",
        },
        "decisions": {
            "block_size": "16 = vLLM default KV block size on H100 (FlashAttention).",
            "max_num_seqs": "1024 and max_num_batched_tokens 8192: vLLM V1 OpenAI-server defaults"
            " for >=70 GiB GPUs (hypothesis from vLLM source knowledge; capacity is unchanged"
            " between 256 and 1024 per the estimator).",
            "max_model_len": "131072 per operator (YaRN).",
        },
        "versions": {
            "aisimulate": importlib.metadata.version("aisimulate"),
            "ai-dynamo-runtime": importlib.metadata.version("ai-dynamo-runtime"),
        },
    }
    out = CR / "config" / "engine.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(engine, indent=2, sort_keys=True) + "\n"
    out.write_text(text)
    print(out, hashlib.sha256(text.encode()).hexdigest(), num_gpu_blocks)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import model_info
from pydantic import BaseModel
from transformers import AutoConfig

DTYPE_BYTES_MAP = {
    "F32": 4,  # FP32: 4 bytes per parameter
    "BF16": 2,  # BF16: 2 bytes per parameter
    "F16": 2,  # FP16: 2 bytes per parameter
    "F8_E4M3": 1,  # FP8: 1 byte per parameter
    "F8_E5M2": 1,  # FP8: 1 byte per parameter
    "F8_E8M0": 1,  # FP8: 1 byte per parameter
    "I8": 1,  # INT8: 1 byte per parameter
    "I4": 0.5,  # INT4: 0.5 bytes per parameter
}

CONTEXT_LENGTH_ATTRS = [
    "max_position_embeddings",  # Most common (BERT, GPT, LLaMA, etc.)
    "n_positions",  # GPT-2, GPT-Neo
    "max_sequence_length",  # Some models
    "seq_length",  # Some older models
    "model_max_length",  # Some tokenizer configs
    "sliding_window",  # Mistral with sliding window attention
]

# supported MoE architectures
MOE_ARCHITECTURES = {
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",  # MLA + MoE
    "Qwen3MoeForCausalLM",  # GQA + MoE
}
# MoE architectures that sweeps TP additionally to TEP/DEP
MOE_ADDITIONAL_TP_ARCHITECTURES = {
    "Qwen3MoeForCausalLM",  # GQA + MoE
}
# MoE architectures with Multi-Latent Attention (MLA)
MLA_MOE_ARCHITECTURES = {
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
}

# Minimum GPU memory overhead factor: VRAM must exceed model weight by this
# multiplier to leave room for KV cache, activations, etc.  Mirrors the 1.5×
# rule used in AIC's ``_calculate_min_tp``.
_VRAM_OVERHEAD_FACTOR = 1.5


def calculate_min_gpus_for_model(
    model_size_mb: float,
    vram_per_gpu_mb: float,
) -> int:
    """Return the minimum number of GPUs needed to fit a model.

    Uses the 1.5× rule: the combined VRAM of the returned GPU count must be at
    least ``_VRAM_OVERHEAD_FACTOR`` × ``model_size_mb``.

    The result is always a power of two (1, 2, 4, 8 …) so that it maps cleanly
    to TP / TEP / DEP sizes.
    """
    if vram_per_gpu_mb <= 0:
        raise ValueError(f"vram_per_gpu_mb must be positive, got {vram_per_gpu_mb}")
    if model_size_mb <= 0:
        raise ValueError(f"model_size_mb must be positive, got {model_size_mb}")

    required_vram = model_size_mb * _VRAM_OVERHEAD_FACTOR
    raw_gpus = required_vram / vram_per_gpu_mb
    # Round up to the next power of two
    if raw_gpus <= 1:
        return 1
    return int(2 ** math.ceil(math.log2(raw_gpus)))


def get_default_parallelization_for_architecture(
    architecture: str,
    is_moe: bool,
    num_gpus: int,
    optimization_type: str | None = None,
) -> tuple:
    """Return ``(prefill_mapping, decode_mapping)`` for naive / optimization-type routes.

    The mapping rules are identical for aggregated and disaggregated serving.
    In aggregated mode the caller uses both mappings identically (single
    engine); in disaggregated mode they map to separate prefill/decode workers.

    Rules:
        * **Dense** (``is_moe=False``): TP
        * **MLA + MoE + throughput** (DeepSeek-V3 family): DEP
        * **All other sparse** (MLA + MoE + latency, GQA + MoE either): TEP

    Parameters
    ----------
    architecture:
        HuggingFace model architecture class name.
    is_moe:
        Whether the model is a Mixture-of-Experts model.
    num_gpus:
        Number of GPUs to distribute across.
    optimization_type:
        ``"throughput"`` or ``"latency"`` (or *None* for the default/SLA path
        which falls back to TEP for MoE models).

    Returns
    -------
    tuple[dict, dict]
        ``(prefill_mapping, decode_mapping)`` where each mapping is a dict
        with exactly one key from ``{"tp", "tep", "dep"}``.
    """
    if not is_moe:
        m = {"tp": num_gpus}
        return m, m

    # MLA + MoE + throughput → DEP (maximises concurrent KV-cache utilisation)
    if (
        architecture in MLA_MOE_ARCHITECTURES
        and optimization_type == "throughput"
    ):
        m = {"dep": num_gpus}
        return m, m

    # All other sparse (MLA+MoE+latency, GQA+MoE either) → TEP
    m = {"tep": num_gpus}
    return m, m


def get_local_model_weight_size(
    model_path: Union[str, Path],
) -> float:
    """Return model size in MB by scanning local directory."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")

    # Weight file extensions to look for
    weight_extensions = [".safetensors", ".bin", ".pt", ".pth"]

    total_size_bytes = 0
    for file_path in model_path.rglob("*"):
        if file_path.is_file() and any(
            str(file_path).endswith(ext) for ext in weight_extensions
        ):
            total_size_bytes += file_path.stat().st_size

    return total_size_bytes / (1024**2)


def get_model_weight_size_from_hub(
    model_name: str,
    token: Optional[str] = None,
) -> float:
    """Return model size in MB by querying Hugging Face Hub API."""
    try:
        info = model_info(model_name, token=token)

        # Filter for model weight files (safetensors or pytorch bin files)
        # Also filter out files with None size
        weight_extensions = [".safetensors", ".bin", ".pt", ".pth"]
        total_size_bytes = 0

        if info.siblings is not None:
            for sibling in info.siblings:
                if any(sibling.rfilename.endswith(ext) for ext in weight_extensions):
                    if sibling.size is not None:
                        total_size_bytes += sibling.size

        # If no file sizes were available, try to estimate from safetensors metadata
        if total_size_bytes == 0 and info.safetensors is not None:
            # SafeTensors info gives parameter counts per dtype
            for dtype, param_count in info.safetensors.parameters.items():
                bytes_per_param = DTYPE_BYTES_MAP.get(
                    dtype, 2
                )  # Default to 2 bytes (FP16/BF16)
                total_size_bytes += int(param_count * bytes_per_param)

        return total_size_bytes / (1024**2)
    except Exception as e:
        raise RuntimeError(f"Failed to get model info from Hub: {e}")


def get_model_weight_size(
    model_name_or_path: Union[str, Path],
) -> float:
    """Return model size in MB (auto-detects local vs HF Hub)."""
    path = Path(model_name_or_path)

    if path.exists() and path.is_dir():
        # Local model
        return get_local_model_weight_size(model_name_or_path)
    else:
        # HF Hub model
        return get_model_weight_size_from_hub(str(model_name_or_path))


class ModelInfo(BaseModel):
    model_size: float
    architecture: str
    is_moe: bool
    max_context_length: Optional[int] = None
    num_experts: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_kv_heads: Optional[int] = None
    quantization_block_size: Optional[int] = None


def get_model_info(
    model_name_or_path: Union[str, Path],
    trust_remote_code: bool = False,
) -> ModelInfo:
    model_size = get_model_weight_size(model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    architecture = config.architectures[0]
    if architecture in MOE_ARCHITECTURES:
        is_moe = True
    else:
        is_moe = False

    # Detect max context length from config
    # Different models use different attribute names for max context length
    max_context_length = None
    for attr in CONTEXT_LENGTH_ATTRS:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                max_context_length = value
                break

    # Detect number of experts for MoE models
    # Different models use different attribute names
    num_experts = None
    if is_moe:
        expert_attrs = [
            "n_routed_experts",  # DeepSeek V3/R1
            "num_local_experts",  # Mixtral, Qwen
            "num_experts",  # Generic
        ]
        for attr in expert_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                if value is not None:
                    num_experts = value
                    break

    # Detect intermediate size (FFN hidden dimension)
    intermediate_size = None
    intermediate_attrs = [
        "intermediate_size",  # Most common (BERT, LLaMA, etc.)
        "ffn_dim",  # Some transformer models
    ]
    for attr in intermediate_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                intermediate_size = value
                break

    # Detect number of key-value heads (for GQA)
    num_kv_heads = None
    kv_head_attrs = [
        "num_key_value_heads",  # LLaMA 2/3, Mistral, etc.
        "num_kv_heads",  # Alternative name
    ]
    for attr in kv_head_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                num_kv_heads = value
                break
    # If not found, check if it equals num_attention_heads (standard MHA)
    if num_kv_heads is None and hasattr(config, "num_attention_heads"):
        num_kv_heads = config.num_attention_heads

    # Detect quantization block size
    quantization_block_size = None
    if hasattr(config, "quantization_config"):
        quant_config = config.quantization_config
        if isinstance(quant_config, dict):
            # Check for common quantization block size attributes
            quantization_block_size = (
                quant_config.get("weight_block_size")
                or quant_config.get("block_size")
                or quant_config.get("group_size")
                or quant_config.get("q_group_size")
            )
        elif quant_config is not None:
            # Handle object-based quantization config
            for attr in [
                "weight_block_size",
                "block_size",
                "group_size",
                "q_group_size",
            ]:
                if hasattr(quant_config, attr):
                    value = getattr(quant_config, attr)
                    if value is not None:
                        quantization_block_size = value
                        break

        # Handle case where block size is a list (e.g., [128, 128] for [input, output] block sizes)
        if (
            isinstance(quantization_block_size, list)
            and len(quantization_block_size) > 0
        ):
            quantization_block_size = max(quantization_block_size)

    return ModelInfo(
        model_size=model_size,
        architecture=architecture,
        is_moe=is_moe,
        max_context_length=max_context_length,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        num_kv_heads=num_kv_heads,
        quantization_block_size=quantization_block_size,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    print(get_model_info(args.model))

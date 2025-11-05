# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Union

from huggingface_hub import model_info
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

# only for MLA + MoE models, treat other MoE models as dense models
MOE_ARCHITECTURES = {"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}


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


def get_model_info(
    model_name_or_path: Union[str, Path],
    trust_remote_code: bool = False,
) -> dict:
    model_size = get_model_weight_size(model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    if config.architectures[0] in MOE_ARCHITECTURES:
        config.is_moe = True
    else:
        config.is_moe = False

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
    if config.is_moe:
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

    return {
        "model_size": model_size,
        "is_moe": config.is_moe,
        "max_context_length": max_context_length,
        "num_experts": num_experts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    print(get_model_info(args.model))

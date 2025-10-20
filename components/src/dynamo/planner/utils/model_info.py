# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union, Optional

from transformers import AutoConfig, PretrainedConfig
from huggingface_hub import model_info


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
    weight_extensions = ['.safetensors', '.bin', '.pt', '.pth']
    
    total_size_bytes = 0
    for file_path in model_path.rglob('*'):
        if file_path.is_file() and any(str(file_path).endswith(ext) for ext in weight_extensions):
            total_size_bytes += file_path.stat().st_size
    
    return total_size_bytes / (1024 ** 2)

def get_model_weight_size_from_hub(
    model_name: str,
    token: Optional[str] = None,
) -> float:
    """Return model size in MB by querying Hugging Face Hub API."""
    try:
        info = model_info(model_name, token=token)
        
        # Filter for model weight files (safetensors or pytorch bin files)
        # Also filter out files with None size
        weight_extensions = ['.safetensors', '.bin', '.pt', '.pth']
        total_size_bytes = 0
        
        for sibling in info.siblings:
            if any(sibling.rfilename.endswith(ext) for ext in weight_extensions):
                if sibling.size is not None:
                    total_size_bytes += sibling.size
        
        # If no file sizes were available, try to estimate from safetensors metadata
        if total_size_bytes == 0 and info.safetensors is not None:
            # SafeTensors info gives us the total parameter count
            # Estimate size based on the dtype
            total_params = info.safetensors.total
            
            # Check the dtype from parameters dict
            if 'BF16' in info.safetensors.parameters or 'F16' in info.safetensors.parameters:
                bytes_per_param = 2  # BF16/FP16 uses 2 bytes per parameter
            elif 'F32' in info.safetensors.parameters:
                bytes_per_param = 4  # FP32 uses 4 bytes per parameter
            elif 'I8' in info.safetensors.parameters:
                bytes_per_param = 1  # INT8 uses 1 byte per parameter
            else:
                bytes_per_param = 2  # Default to FP16/BF16
            
            total_size_bytes = total_params * bytes_per_param
        
        return total_size_bytes / (1024 ** 2)
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
        return get_model_weight_size_from_hub(model_name_or_path)


def get_model_info(
    model_name_or_path: Union[str, Path],
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    model_size = get_model_weight_size(model_name_or_path)

    import pdb; pdb.set_trace()
    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    config = get_model_info(args.model)
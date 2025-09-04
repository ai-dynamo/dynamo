#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Using Speculative Decoding with vLLM Backend in Dynamo

This example demonstrates how to use the new speculative decoding support
in the vLLM backend with YAML configuration files.

Usage:
    python speculative_decoding_example.py

Prerequisites:
    - Dynamo environment activated
    - vLLM backend dependencies installed
    - Access to draft models (if using Eagle)
"""

import tempfile
import yaml
import os

def create_example_configs():
    """Create example configuration files for different speculative decoding modes."""
    
    # Eagle configuration
    eagle_config = {
        "tensor_parallel_size": 1,
        "max_batch_size": 64,
        "max_num_tokens": 512,
        "max_seq_len": 2048,
        "gpu_memory_utilization": 0.8,
        "speculative_config": {
            "decoding_type": "Eagle",
            "max_draft_len": 3,
            "speculative_model_dir": "facebook/opt-125m",  # Small model for demo
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.8
        },
        "enable_prefix_caching": True,
        "block_size": 16
    }
    
    # MTP configuration (maps to n-gram)
    mtp_config = {
        "tensor_parallel_size": 1,
        "max_batch_size": 64,
        "max_num_tokens": 512,
        "max_seq_len": 2048,
        "gpu_memory_utilization": 0.8,
        "speculative_config": {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": 1,
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.8
        },
        "enable_prefix_caching": True,
        "block_size": 16
    }
    
    # Write configs to temporary files
    eagle_path = "/tmp/eagle_config.yaml"
    mtp_path = "/tmp/mtp_config.yaml"
    
    with open(eagle_path, 'w') as f:
        yaml.dump(eagle_config, f, default_flow_style=False)
    
    with open(mtp_path, 'w') as f:
        yaml.dump(mtp_config, f, default_flow_style=False)
    
    return eagle_path, mtp_path

def main():
    """Main example function."""
    print("🚀 Dynamo vLLM Speculative Decoding Example")
    print("=" * 50)
    
    # Create example configurations
    eagle_path, mtp_path = create_example_configs()
    
    print("\n📋 Example Commands:")
    print("\n1. Eagle Speculative Decoding (Draft Model):")
    print(f"   python -m dynamo.vllm \\")
    print(f"       --model facebook/opt-6.7b \\")
    print(f"       --extra-engine-args {eagle_path}")
    
    print("\n2. MTP Speculative Decoding (N-gram Lookup):")
    print(f"   python -m dynamo.vllm \\")
    print(f"       --model facebook/opt-6.7b \\")
    print(f"       --extra-engine-args {mtp_path}")
    
    print("\n3. Disaggregated Serving with Speculative Decoding:")
    print("   # Prefill worker")
    print(f"   python -m dynamo.vllm \\")
    print(f"       --model facebook/opt-6.7b \\")
    print(f"       --is-prefill-worker \\")
    print(f"       --extra-engine-args {eagle_path}")
    print("\n   # Decode worker")
    print(f"   python -m dynamo.vllm \\")
    print(f"       --model facebook/opt-6.7b \\")
    print(f"       --extra-engine-args {eagle_path}")
    
    print("\n📄 Configuration Files Created:")
    print(f"   Eagle config: {eagle_path}")
    print(f"   MTP config: {mtp_path}")
    
    print("\n📖 View configuration contents:")
    print(f"   cat {eagle_path}")
    print(f"   cat {mtp_path}")
    
    print("\n✨ Features:")
    print("   • Compatible with TRT-LLM YAML configuration format")
    print("   • Automatic conversion of Eagle → Draft Model speculative decoding")
    print("   • Automatic conversion of MTP → N-gram lookup speculative decoding")
    print("   • Seamless integration with existing Dynamo workflows")
    
    print(f"\n📚 For more information, see: components/backends/vllm/SPECULATIVE_DECODING.md")

if __name__ == "__main__":
    main() 
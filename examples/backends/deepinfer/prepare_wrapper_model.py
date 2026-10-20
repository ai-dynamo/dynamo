# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download only tokenizer and configuration assets, never model weights."""

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print(
        snapshot_download(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            revision="314113f8a7dadb388215396ba5fbee2c610a04d6",
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the TensorRT-LLM diffusion backend.

Usage:
    python -m dynamo.trtllm.unified_diffusion_main --modality image_diffusion <args>
    python -m dynamo.trtllm.unified_diffusion_main --modality video_diffusion <args>

The token-pipeline LLM entry point is dynamo.trtllm.unified_main.
"""

from dynamo.common.backend.run import run
from dynamo.trtllm.unified_diffusion import TrtllmDiffusionEngine


def main():
    run(TrtllmDiffusionEngine)


if __name__ == "__main__":
    main()

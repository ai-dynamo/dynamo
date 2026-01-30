# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running TensorRT-LLM Video Diffusion worker.

Usage:
    python -m dynamo.trtllm_diffusion --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
"""

import uvloop

from dynamo.trtllm_diffusion.main import worker

if __name__ == "__main__":
    uvloop.run(worker())

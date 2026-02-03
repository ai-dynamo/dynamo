# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running TensorRT-LLM Video Diffusion worker.

DEPRECATED: This module is deprecated. Use dynamo.trtllm with --modality video_diffusion instead.

Usage (deprecated):
    python -m dynamo.trtllm_diffusion --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...

New usage:
    python -m dynamo.trtllm --modality video_diffusion --model-type wan_t2v --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
"""

import sys
import warnings

# Show deprecation warning
warnings.warn(
    "\n"
    "dynamo.trtllm_diffusion is DEPRECATED and will be removed in a future release.\n"
    "\n"
    "Please use dynamo.trtllm with --modality video_diffusion instead:\n"
    "  python -m dynamo.trtllm --modality video_diffusion --model-type wan_t2v --model-path ...\n"
    "\n"
    "This module will continue to work but is no longer maintained.\n",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    # Continue with the original behavior for backward compatibility
    import uvloop

    from dynamo.trtllm_diffusion.main import worker

    uvloop.run(worker())

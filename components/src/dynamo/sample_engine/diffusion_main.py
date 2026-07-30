# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample diffusion backend (CPU-only).

Usage:
    python -m dynamo.sample_engine.diffusion_main --model-name sample-diffusion-model
"""

from dynamo.backend import run

from .diffusion_engine import SampleDiffusionEngine


def main():
    run(SampleDiffusionEngine)


if __name__ == "__main__":
    main()

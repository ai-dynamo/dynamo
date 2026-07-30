# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample backend.

Usage:
    python -m dynamo.sample_engine --model-name test-model
"""

from dynamo.backend import run

from .engine import SampleLLMEngine


def main():
    run(SampleLLMEngine)


if __name__ == "__main__":
    main()

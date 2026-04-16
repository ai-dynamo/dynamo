# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the SGLang backend.

Usage:
    python -m dynamo.sglang.unified_main <sglang args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

from dynamo.common.backend.run import run
from dynamo.sglang.llm_engine import SglangLLMEngine


def main():
    run(SglangLLMEngine)


if __name__ == "__main__":
    main()

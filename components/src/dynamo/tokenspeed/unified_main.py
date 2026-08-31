# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the TokenSpeed backend.

Usage:
    python -m dynamo.tokenspeed <tokenspeed args>
"""

from dynamo.common.backend.run import run
from dynamo.tokenspeed.llm_engine import TokenspeedLLMEngine


def main():
    run(TokenspeedLLMEngine)


if __name__ == "__main__":
    main()

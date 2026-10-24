# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the DeepInfer backend.

Usage:
    python -m dynamo.deepinfer --model PATH [--dp-size N] [DeepInfer and Dynamo args]
"""

from dynamo.common.backend.run import run
from dynamo.deepinfer.engine import DeepInferEngine


def main() -> None:
    run(DeepInferEngine)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample multimodal (VLM) backend.

Usage:
    python -m dynamo.common.backend.sample_multimodal_main --model-name my-vlm
    python -m dynamo.common.backend.sample_multimodal_main --disaggregation-mode encode
"""

from dynamo.common.backend.run import run
from dynamo.common.backend.sample_multimodal_engine import SampleMultimodalEngine


def main():
    run(SampleMultimodalEngine)


if __name__ == "__main__":
    main()

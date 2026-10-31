# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.common.backend.run import run
from dynamo.deepinfer.llm_engine import DeepInferLLMEngine


def main() -> None:
    run(DeepInferLLMEngine)


if __name__ == "__main__":
    main()

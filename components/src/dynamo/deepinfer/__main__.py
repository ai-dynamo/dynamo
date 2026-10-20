# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import traceback

from dynamo.common.backend.run import run
from dynamo.deepinfer.args import parse_args
from dynamo.deepinfer.launch import launch_ranks
from dynamo.deepinfer.llm_engine import DeepInferLLMEngine

if __name__ == "__main__":
    config = parse_args()
    if config.engine_args.get("data_parallel_size", 1) > 1 and not config.rank_worker:
        raise SystemExit(launch_ranks(config))
    try:
        run(DeepInferLLMEngine)
    except Exception:
        if not config.rank_worker:
            raise
        traceback.print_exc()
        # Native child-process finalizers can hang after a failed initialization.
        os._exit(1)
    if config.rank_worker:
        os._exit(0)

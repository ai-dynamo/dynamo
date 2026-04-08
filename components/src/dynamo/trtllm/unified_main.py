# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the TensorRT-LLM backend using DynamoPythonBackendModel.

Usage:
    python -m dynamo.trtllm.unified_main <trtllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

import uvloop
from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from torch.cuda import device_count

from dynamo.common.backend import BackendConfig, DynamoPythonBackendModel
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.dynamo_engine import TrtllmDynamoEngine
from dynamo.trtllm.engine import Backend


async def worker():
    config = parse_args()

    gpus_per_node = config.gpus_per_node or device_count()

    engine_args = {
        "model": str(config.model),
        "scheduler_config": SchedulerConfig(),
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "backend": Backend.PYTORCH,
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=config.free_gpu_memory_fraction,
        ),
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
    }

    engine = TrtllmDynamoEngine(
        engine_args=engine_args,
        model_name=config.model,
        served_model_name=config.served_model_name,
        max_seq_len=config.max_seq_len,
        max_batch_size=config.max_batch_size,
        max_num_tokens=config.max_num_tokens,
        kv_block_size=config.kv_block_size,
    )
    backend_config = BackendConfig.from_runtime_config(
        config,
        model_name=config.model,
        served_model_name=config.served_model_name,
        model_input=ModelInput.Tokens,
    )
    model = DynamoPythonBackendModel(backend_config, engine)
    await model.run()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()

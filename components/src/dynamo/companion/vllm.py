# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific logic for companion server model loading."""

import hashlib
import logging
import pickle
import copy
import torch

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.utils import get_distributed_init_method

logger = logging.getLogger(__name__)


class VllmModelLoader:
    """Handles vLLM-specific model loading for the companion server."""

    def __init__(
        self,
        config: VllmConfig,
        device_id: int,
        local_rank: int,
        global_rank: int,
        world_size: int,
        companion_master_port: int = 29500,
    ):
        self.config = copy.deepcopy(config)
        self.config.load_config.enable_companion_process = False
        self.config.model_config.enforce_eager = True

        self.device_id = device_id
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.companion_master_port = companion_master_port

    def compute_config_hash(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        global_rank: int,
        world_size: int
    ) -> str:
        """Compute a hash for the model configuration."""
        config_data = {
            'model_config_hash': vllm_config.model_config.compute_hash(),
            'parallel_config_hash': vllm_config.parallel_config.compute_hash(),
            'device_config_hash': vllm_config.device_config.compute_hash(),
            'load_config_hash': vllm_config.load_config.compute_hash(),
            'lora_config_hash': (
                vllm_config.lora_config.compute_hash()
                if vllm_config.lora_config else "None"
            ),
            'local_rank': local_rank,
            'global_rank': global_rank,
            'world_size': world_size
        }
        return hashlib.md5(pickle.dumps(config_data)).hexdigest()

    def load_model_weights(self) -> torch.nn.Module:
        """Load model weights with distributed initialization."""
        logger.info("Starting model weight loading")

        # Set the current CUDA device for this process
        # This must be done before any model loading to ensure tensors are
        # created on the correct GPU
        logger.info(f"Setting CUDA device to {self.device_id}")
        torch.cuda.set_device(self.device_id)

        # Verify the device is set correctly
        current_device = torch.cuda.current_device()
        assert current_device == self.device_id, (
            f"Failed to set CUDA device: expected {self.device_id}, "
            f"got {current_device}"
        )
        logger.info(f"Current CUDA device verified: {current_device}")

        # Clear any existing layer registrations to avoid duplicate layer name errors
        self.config.compilation_config.static_forward_context.clear()
        logger.info("Cleared static_forward_context for fresh model load")

        # Get distributed init method
        init_method = get_distributed_init_method(
            self.config.parallel_config.data_parallel_master_ip,
            self.companion_master_port
        )

        # Initialize distributed environment using the same pattern as workers
        # Use 'gloo' backend for fake/mock distributed initialization
        logger.info(
            f"Initializing distributed environment: global rank={self.global_rank}, "
            f"local rank={self.local_rank}, world size={self.world_size}, init_method={init_method}"
        )

        init_worker_distributed_environment(
            vllm_config=self.config,
            rank=self.global_rank,
            distributed_init_method=init_method,
            local_rank=self.local_rank,
            backend="gloo"  # Use gloo for mock distributed
        )

        logger.info("Distributed environment initialized successfully")

        # Create model loader
        assert self.config.load_config is not None
        logger.info(
            f"Creating DefaultModelLoader with device_type={self.config.device_config.device_type}, "
            f"current_device={torch.cuda.current_device()}, "
            f"load_format={self.config.load_config.load_format}"
        )

        default_loader = DefaultModelLoader(self.config.load_config)

        logger.info(f"Loading model {self.config.model_config.model} on CUDA device {self.device_id}")
        model = default_loader.load_model(
            self.config, self.config.model_config,
        )
        logger.info(f"Model loaded successfully on device {torch.cuda.current_device()}!")

        return model

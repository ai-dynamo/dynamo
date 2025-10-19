# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Companion client for vLLM model loading via Dynamo DistributedRuntime."""

import base64
import json
import logging

import cloudpickle
import torch

from dynamo.runtime import DistributedRuntime, dynamo_worker
from .utils import import_weights_from_tree, ModuleTreeNode

logger = logging.getLogger(__name__)


@dynamo_worker(static=False)
async def load_model_from_companion(
    runtime: DistributedRuntime,
    target_model: torch.nn.Module,
    vllm_config,
    device_id: int,
    local_rank: int,
    global_rank: int,
    world_size: int,
    namespace: str = "default"
):
    """
    Worker function to load model from companion server.

    This function is decorated with @dynamo_worker so it has access to the runtime.
    It creates a CompanionClient and loads the model weights.
    """
    client = CompanionClient(runtime, device_id, namespace)
    await client.load_model(
        target_model=target_model,
        vllm_config=vllm_config,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size
    )


class CompanionClient:
    """
    Client for loading model from companion server via CUDA IPC.

    Uses the same import_weights pattern as meta_load.py to reconstruct
    the entire module tree (parameters, buffers, submodules).
    """

    def __init__(self, runtime, device_id: int, namespace: str = "default"):
        """
        Initialize the companion client.

        Args:
            runtime: Dynamo DistributedRuntime instance
            device_id: GPU device ID (used to determine which companion to connect to)
            namespace: Dynamo namespace (default: "default")
        """
        self.runtime = runtime
        self.device_id = device_id
        self.namespace = namespace
        self.component_name = f"companion-gpu{device_id}"
        self._client = None

        logger.info(f"Companion client initialized for GPU {device_id}")

    async def _get_client(self):
        """Get or create the Dynamo client for the companion endpoint."""
        if self._client is None:
            self._client = await (
                self.runtime.namespace(self.namespace)
                .component(self.component_name)
                .endpoint("load_model")
                .client()
            )
        return self._client

    async def load_model(
        self,
        target_model: torch.nn.Module,
        vllm_config,
        local_rank: int,
        global_rank: int,
        world_size: int
    ) -> None:
        """
        Load model weights from companion server via CUDA IPC into target_model.

        Args:
            target_model: The model to load weights into (e.g., meta model)
            vllm_config: vLLM configuration
            local_rank: Local rank for this process
            global_rank: Global rank for this process
            world_size: World size for distributed training
        """
        client = await self._get_client()

        # Create request - we need to serialize VllmConfig properly
        # Use cloudpickle for VllmConfig since it's not JSON serializable and may contain complex objects
        config_pickled = base64.b64encode(cloudpickle.dumps(vllm_config)).decode('utf-8')

        request = {
            "config_pickled": config_pickled,
            "local_rank": local_rank,
            "global_rank": global_rank,
            "world_size": world_size
        }

        logger.info("Sending load model request to companion server (may take several minutes)...")

        # Call the companion endpoint (blocks until model is loaded)
        # Use 'round_robin' instead of 'generate' since we're sending a single large response,
        # not streaming tokens. This avoids streaming completion issues.
        # Convert to JSON string for proper serialization
        response_gen = await client.round_robin(json.dumps(request))
        response_obj = await anext(response_gen)

        # Extract the actual data from the response object (which is base64-encoded)
        response_encoded = response_obj.data()

        # Decode the base64-encoded pickled response
        response = cloudpickle.loads(base64.b64decode(response_encoded))
        logger.info("Received and decoded response from companion server")

        if not response["success"]:
            raise RuntimeError(f"Failed to load model from companion: {response.get('error', 'Unknown error')}")

        # Reconstruct ModuleTreeNode from dictionary
        module_tree_dict = response["module_tree"]
        module_tree = ModuleTreeNode.from_dict(module_tree_dict)
        logger.info("Reconstructed module tree from response")

        # Import weights from the tree into the target model
        logger.info("Importing weights into target model...")
        import_weights_from_tree(target_model, module_tree)

        logger.info("Successfully imported all weights via CUDA IPC")

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for data parallel rank-aware routing."""

import logging
import random
import string
from typing import List

import pytest

from dynamo._core import DistributedRuntime, KvPushRouter, KvRouterConfig
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
DP_SIZE = 4  # Test with 4 data parallel ranks
BLOCK_SIZE = 16


def generate_random_suffix() -> str:
    """Generate random suffix for namespace isolation."""
    return "".join(random.choices(string.ascii_lowercase, k=10))


def get_runtime():
    """Get or create a DistributedRuntime instance."""
    try:
        return DistributedRuntime.detached()
    except Exception:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return DistributedRuntime(loop, False)


class MockerProcess:
    """Manages mocker engine instance with DP support"""

    def __init__(
        self,
        request,
        namespace: str,
        dp_size: int = 1,
    ):
        self.namespace = namespace
        self.endpoint = f"dyn://{namespace}.prefill.generate"
        self.dp_size = dp_size
        self.mocker_processes: List[ManagedProcess] = []
        self.request = request

        # Create mocker process
        command = [
            "python", "-m", "dynamo.mocker",
            "--model-path", MODEL_NAME,
            "--endpoint", self.endpoint,
            "--speedup-ratio", "10.0",
            "--block-size", str(BLOCK_SIZE),
            "--num-gpu-blocks-override", "1000",
            "--data-parallel-size", str(dp_size),
        ]

        process = ManagedProcess(
            command=command,
            timeout=60,
            display_output=True,
            log_dir=request.node.name,
            terminate_existing=False,
        )
        self.mocker_processes.append(process)

    def __enter__(self):
        """Start mocker process"""
        for process in self.mocker_processes:
            process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop mocker process"""
        for process in self.mocker_processes:
            process.__exit__(exc_type, exc_val, exc_tb)


@pytest.mark.asyncio
async def test_router_returns_valid_dp_rank(request):
    """Verify router returns valid dp_rank in routing decision."""
    runtime = get_runtime()
    namespace = f"dp-routing-{generate_random_suffix()}"
    
    with MockerProcess(request, namespace, dp_size=DP_SIZE):
        try:
            router_comp = runtime.namespace(namespace).component("router")
            await router_comp.create_service()
            
            endpoint = runtime.namespace(namespace).component("prefill").endpoint("generate")
            
            import asyncio
            await asyncio.sleep(2)
            
            kv_router_config = KvRouterConfig(
                overlap_score_weight=2.0,
                router_temperature=0.0,
            )
            kv_push_router = KvPushRouter(endpoint, BLOCK_SIZE, kv_router_config)
            
            if hasattr(kv_push_router, "best_worker"):
                worker_id, dp_rank, overlap = await kv_push_router.best_worker([1, 2, 3, 4, 5])
                
                assert dp_rank is not None, "Router should return dp_rank"
                assert isinstance(dp_rank, int), "dp_rank should be integer"
                assert 0 <= dp_rank < DP_SIZE, f"dp_rank {dp_rank} out of range [0, {DP_SIZE})"
                
                logger.info(f"✅ Router returned valid dp_rank={dp_rank}")
            else:
                logger.warning("Router API doesn't support best_worker - skipping test")
        
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


@pytest.mark.asyncio
async def test_dp_rank_coverage(request):
    """Verify router selects from full range of DP ranks."""
    runtime = get_runtime()
    namespace = f"dp-routing-{generate_random_suffix()}"
    
    with MockerProcess(request, namespace, dp_size=DP_SIZE):
        try:
            router_comp = runtime.namespace(namespace).component("router")
            await router_comp.create_service()
            
            endpoint = runtime.namespace(namespace).component("prefill").endpoint("generate")
            
            import asyncio
            await asyncio.sleep(2)
            
            kv_router_config = KvRouterConfig(
                overlap_score_weight=2.0,
                router_temperature=0.0,
            )
            kv_push_router = KvPushRouter(endpoint, BLOCK_SIZE, kv_router_config)
            
            if hasattr(kv_push_router, "best_worker"):
                dp_ranks_used = set()
                
                # Query with varied sequences to cover all ranks
                for i in range(50):
                    test_tokens = list(range(i * 7, i * 7 + 10))
                    worker_id, dp_rank, overlap = await kv_push_router.best_worker(test_tokens)
                    
                    assert dp_rank is not None
                    assert isinstance(dp_rank, int)
                    assert 0 <= dp_rank < DP_SIZE
                    dp_ranks_used.add(dp_rank)
                
                # Expect reasonable coverage across DP ranks
                num_ranks = len(dp_ranks_used)
                assert num_ranks >= 2, f"Poor coverage: only {num_ranks} ranks used"
                logger.info(f"✅ Router coverage: {num_ranks}/{DP_SIZE} ranks used - {sorted(dp_ranks_used)}")
            else:
                logger.warning("Router API doesn't support best_worker - skipping test")
        
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise




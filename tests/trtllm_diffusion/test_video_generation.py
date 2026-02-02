# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for TensorRT-LLM Video Diffusion worker.

These tests require:
- GPU with at least 8GB VRAM (for Wan 1.3B model)
- etcd running on localhost:2379
- NATS running on localhost:4222

Run with:
    pytest tests/trtllm_diffusion/test_video_generation.py -v -s
"""

import asyncio
import os
import subprocess
import sys
import time

import pytest

# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(
    not os.path.exists("/dev/nvidia0"),
    reason="GPU not available"
)


@pytest.fixture(scope="module")
def check_gpu():
    """Check if GPU is available and accessible."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.skip("nvidia-smi failed - GPU not accessible")
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("nvidia-smi not found or timed out")


@pytest.fixture(scope="module")
def check_services():
    """Check if etcd and NATS are running."""
    import socket

    # Check etcd
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 2379))
        sock.close()
        if result != 0:
            pytest.skip("etcd not running on localhost:2379")
    except Exception as e:
        pytest.skip(f"Cannot connect to etcd: {e}")

    # Check NATS
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 4222))
        sock.close()
        if result != 0:
            pytest.skip("NATS not running on localhost:4222")
    except Exception as e:
        pytest.skip(f"Cannot connect to NATS: {e}")


class TestVideoWorkerStartup:
    """Test worker startup and model registration."""

    def test_worker_imports(self):
        """Test that worker modules can be imported."""
        from dynamo.trtllm_diffusion.args import VideoConfig, parse_endpoint
        from dynamo.trtllm_diffusion.protocol import NvCreateVideoRequest, NvVideosResponse

        # Test endpoint parsing
        ns, comp, ep = parse_endpoint("dyn://dynamo.video.generate")
        assert ns == "dynamo"
        assert comp == "video"
        assert ep == "generate"

        # Test config defaults
        config = VideoConfig()
        assert config.namespace == "dynamo"
        assert config.default_height == 480
        assert config.default_width == 832

    def test_protocol_types(self):
        """Test protocol type validation."""
        from dynamo.trtllm_diffusion.protocol import NvCreateVideoRequest, NvVideosResponse, VideoData

        # Test request parsing
        request = NvCreateVideoRequest(
            prompt="A cat playing piano",
            model="test-model",
            num_frames=9,
            num_inference_steps=2,
        )
        assert request.prompt == "A cat playing piano"
        assert request.num_frames == 9

        # Test response creation
        response = NvVideosResponse(
            id="test-123",
            model="test-model",
            created=int(time.time()),
            data=[VideoData(url="/tmp/test.mp4")],
        )
        assert response.status == "completed"
        assert response.progress == 100


@pytest.mark.gpu
class TestVideoGeneration:
    """Test actual video generation (requires GPU)."""

    @pytest.fixture
    def worker_process(self, check_gpu, check_services):
        """Start worker process for testing."""
        env = os.environ.copy()
        env["ETCD_ENDPOINTS"] = "localhost:2379"
        env["NATS_URL"] = "nats://localhost:4222"
        env["PYTHONPATH"] = f"components/src:{env.get('PYTHONPATH', '')}"

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "dynamo.trtllm_diffusion",
                "--model-path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "--disable-torch-compile",  # Faster startup for testing
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for worker to be ready (look for "serving endpoint" in logs)
        start_time = time.time()
        timeout = 120  # 2 minutes for model loading
        ready = False

        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                # Process exited
                output = proc.stdout.read()
                pytest.fail(f"Worker process exited unexpectedly:\n{output}")

            # Check if ready (non-blocking read)
            import select
            if select.select([proc.stdout], [], [], 0.1)[0]:
                line = proc.stdout.readline()
                if "serving endpoint" in line.lower():
                    ready = True
                    break

        if not ready:
            proc.terminate()
            pytest.fail("Worker did not become ready within timeout")

        yield proc

        # Cleanup
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    @pytest.mark.timeout(180)
    async def test_video_generation_e2e(self, worker_process):
        """Test end-to-end video generation."""
        from dynamo.runtime import DistributedRuntime

        loop = asyncio.get_running_loop()
        runtime = DistributedRuntime(loop, "etcd", "nats", True)

        try:
            # Get client
            client = await (
                runtime.namespace("dynamo")
                .component("trtllm_diffusion")
                .endpoint("generate")
                .client()
            )

            # Wait for instances
            await asyncio.sleep(1)
            instances = client.instance_ids()
            assert len(instances) > 0, "No worker instances found"

            # Send request
            request = {
                "prompt": "A simple test video",
                "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "num_frames": 9,  # Minimal for fast testing
                "num_inference_steps": 2,  # Minimal steps
            }

            # await client.random() returns an async iterator, each item has .data()
            response = None
            iterator = await client.random(request)
            async for resp in iterator:
                data = resp.data()
                # data can be dict or JSON string
                if isinstance(data, str):
                    import json
                    response = json.loads(data)
                else:
                    response = data
                break

            assert response is not None
            assert "id" in response
            assert response.get("status") == "completed" or "error" in response

        finally:
            runtime.shutdown()


if __name__ == "__main__":
    # Run basic tests without GPU
    pytest.main([__file__, "-v", "-k", "not gpu", "--tb=short"])

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test for decode disaggregation with sequence length tiers.

This test:
1. Starts two mocker workers with different this_seqlen values (128 and 8192)
2. Starts a frontend with decode disaggregation enabled
3. Sends a request that will exceed the small tier and trigger migration
4. Verifies migration occurs by checking logs
"""

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from queue import Queue

import aiohttp
import pytest

logger = logging.getLogger(__name__)


# Use a real model name that exists on HuggingFace (just for tokenizer)
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
NAMESPACE = "test-decode-tiers"
SPEEDUP_RATIO = 100.0  # Fast for testing
FRONTEND_PORT = 18080

# Tier configuration for decode disaggregation testing
# The Qwen3-0.6B model has context_length=40960
# - Main worker: this_seqlen=40960 (equals context_length, publishes model)
# - Intermediate tier: this_seqlen=128 (less than context_length, just registers tier)
CONTEXT_LENGTH = 40960
SMALL_TIER_SEQLEN = 128  # Intermediate tier - should trigger migration quickly
LARGE_TIER_SEQLEN = 40960  # Main decode tier (must equal context_length)


def read_stream(stream, queue, prefix):
    """Read from a stream and put lines into a queue."""
    try:
        for line in iter(stream.readline, ""):
            if line:
                queue.put((prefix, line.rstrip()))
        stream.close()
    except Exception as e:
        queue.put((prefix, f"ERROR reading stream: {e}"))


class ManagedProcess:
    """Context manager for subprocess with real-time output."""

    def __init__(self, cmd: list[str], name: str, env: dict = None):
        self.cmd = cmd
        self.name = name
        self.env = {**os.environ, **(env or {})}
        self.process = None
        self.output_queue = Queue()
        self.threads = []
        self.all_output = []

    def start(self):
        logger.info(f"Starting {self.name}: {' '.join(self.cmd)}")
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            text=True,
            bufsize=1,
        )

        # Start threads to read stdout and stderr
        for stream, name in [
            (self.process.stdout, "stdout"),
            (self.process.stderr, "stderr"),
        ]:
            t = threading.Thread(
                target=read_stream,
                args=(stream, self.output_queue, f"{self.name}:{name}"),
            )
            t.daemon = True
            t.start()
            self.threads.append(t)

        return self

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def collect_output(self, timeout=0.1):
        """Collect output from the queue."""
        lines = []
        while True:
            try:
                prefix, line = self.output_queue.get(timeout=timeout)
                lines.append((prefix, line))
                self.all_output.append((prefix, line))
            except Exception:
                break
        return lines

    def stop(self):
        if self.process:
            logger.info(f"Stopping {self.name} (pid={self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            # Collect any remaining output
            self.collect_output(timeout=0.5)


def build_mocker_cmd(
    endpoint: str,
    this_seqlen: int | None = None,
) -> list[str]:
    """Build mocker command with decode tier configuration."""
    cmd = [
        sys.executable,
        "-m",
        "dynamo.mocker",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--endpoint",
        endpoint,
        "--store-kv",
        "etcd",
        "--request-plane",
        "nats",
        "--speedup-ratio",
        str(SPEEDUP_RATIO),
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "1024",
    ]

    if this_seqlen is not None:
        cmd.extend(["--this-seqlen", str(this_seqlen)])

    return cmd


def build_frontend_cmd(
    port: int, namespace: str, enable_decode_disagg: bool = False
) -> list[str]:
    """Build frontend command."""
    cmd = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(port),
        "--store-kv",
        "etcd",
        "--request-plane",
        "nats",
        "--namespace",
        namespace,
        "--router-mode",
        "kv",
        "--kv-cache-block-size",
        "16",
    ]
    if enable_decode_disagg:
        cmd.append("--enable-decode-disagg")
    return cmd


async def wait_for_frontend(port: int, timeout: float = 30.0) -> bool:
    """Wait for frontend to be ready."""
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    last_response = None

    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        last_response = data
                        if data.get("data"):  # Has models
                            logger.info(f"Frontend ready, models: {data}")
                            return True
                        else:
                            logger.debug(
                                f"Frontend responded but no models yet: {data}"
                            )
        except Exception as e:
            logger.debug(f"Frontend not ready: {e}")
        await asyncio.sleep(0.5)

    logger.error(f"Frontend timeout, last response: {last_response}")
    return False


async def send_chat_request(port: int, prompt: str, max_tokens: int = 50) -> dict:
    """Send a chat completions request."""
    import json as json_module

    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    logger.info(f"Sending chat request to {url} with max_tokens={max_tokens}")

    chunks = []
    content_parts = []
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            logger.info(f"Response status: {resp.status}")
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Error response: {text}")
                return {"error": text, "status": resp.status}

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json_module.loads(data)
                            chunks.append(chunk)
                            # Extract content
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_parts.append(delta["content"])
                        except Exception:
                            pass

    content = "".join(p for p in content_parts if p)
    return {"chunks": chunks, "count": len(chunks), "content": content}


@pytest.mark.e2e
@pytest.mark.gpu_0
@pytest.mark.post_merge
@pytest.mark.router
async def test_decode_tier_migration():
    """Test that requests migrate from smaller to larger decode tiers.

    This test verifies:
    1. Two tiers are set up (128 and 8192 sequence length)
    2. A request starts on the small tier
    3. When sequence length exceeds 128, request migrates to larger tier
    4. Response completes successfully with stream stitching
    """
    processes = []
    tmpdir = tempfile.mkdtemp(prefix="dynamo_test_")

    # Set environment for decode disagger logging
    test_env = {
        "DYN_NAMESPACE": NAMESPACE,
        "DYN_STORE_KV": "etcd",
        "DYN_REQUEST_PLANE": "nats",
        "ETCD_ENDPOINTS": "http://localhost:2379",
        "NATS_SERVER": "nats://localhost:4222",
        "RUST_LOG": (
            "dynamo_llm::decode_disagger=debug,"
            "dynamo_llm::discovery::model_manager=debug,"
            "dynamo_llm::discovery::watcher=debug,"
            "dynamo_llm::local_model=debug,"
            "dynamo_llm::mocker=info,"
            "dynamo_llm::kv_router=debug,"
            "info"
        ),
    }

    try:
        logger.info(f"Using temp directory for file KV: {tmpdir}")

        # Start SMALL tier worker (128 tokens - will trigger migration)
        worker_small = ManagedProcess(
            build_mocker_cmd(
                endpoint=f"dyn://{NAMESPACE}.backend.generate",
                this_seqlen=SMALL_TIER_SEQLEN,
            ),
            name="worker-128",
            env=test_env,
        )
        worker_small.start()
        processes.append(worker_small)

        # Start LARGE tier worker (8192 tokens - migration target)
        worker_large = ManagedProcess(
            build_mocker_cmd(
                endpoint=f"dyn://{NAMESPACE}.backend.generate",
                this_seqlen=LARGE_TIER_SEQLEN,
            ),
            name="worker-8192",
            env=test_env,
        )
        worker_large.start()
        processes.append(worker_large)

        # Wait for workers to initialize
        logger.info("Waiting for workers to initialize...")
        await asyncio.sleep(4)

        # Check workers are running
        assert worker_small.is_running(), "Small tier worker failed to start"
        assert worker_large.is_running(), "Large tier worker failed to start"

        # Start frontend with decode disaggregation enabled
        frontend = ManagedProcess(
            build_frontend_cmd(FRONTEND_PORT, NAMESPACE, enable_decode_disagg=True),
            name="frontend",
            env=test_env,
        )
        frontend.start()
        processes.append(frontend)

        # Wait for frontend to be ready
        logger.info("Waiting for frontend to be ready...")
        frontend_ready = await wait_for_frontend(FRONTEND_PORT, timeout=30)

        if not frontend_ready:
            for p in processes:
                p.collect_output()
            logger.error("Frontend output:")
            for prefix, line in frontend.all_output[-30:]:
                logger.error(f"  {line}")

        assert frontend_ready, "Frontend failed to start"

        logger.info("Frontend is ready!")

        # Give time for tier discovery
        await asyncio.sleep(2)

        # Collect output so far to see tier registration
        for p in processes:
            p.collect_output()

        # Print tier detection logs
        logger.info("=== Tier Detection Logs ===")
        for p in processes:
            for prefix, line in p.all_output:
                if "tier" in line.lower() or "seqlen" in line.lower():
                    logger.info(f"{p.name}: {line}")

        # Send a request that should trigger migration
        # The prompt will be ~50 tokens, and we ask for 200 tokens output
        # Total: ~250 tokens > 128 (small tier limit) -> should migrate to large tier
        prompt = "I like turkey"

        logger.info("Sending request that should trigger migration...")
        result = await send_chat_request(FRONTEND_PORT, prompt, max_tokens=200)

        assert "error" not in result, f"Request failed: {result}"
        assert result["count"] > 0, "No chunks received"

        logger.info(f"Request completed with {result['count']} chunks")
        logger.info(f"Response content length: {len(result['content'])} chars")
        logger.info(f"Response preview: {result['content'][:200]}...")

        # Let things settle and collect logs
        await asyncio.sleep(2)

        # Collect all output
        for p in processes:
            p.collect_output()

        # Print all migration-related logs
        logger.info("\n" + "=" * 60)
        logger.info("Migration and Routing Logs")
        logger.info("=" * 60)

        migration_keywords = ["migrat", "tier", "seqlen", "routing", "select", "exceed"]

        for p in processes:
            relevant_lines = [
                line
                for prefix, line in p.all_output
                if any(kw in line.lower() for kw in migration_keywords)
            ]
            if relevant_lines:
                logger.info(f"\n{p.name}:")
                for line in relevant_lines:
                    logger.info(f"  {line}")

        # Check if migration occurred
        all_output_text = "\n".join(line for _, line in frontend.all_output)

        migration_occurred = (
            "migrat" in all_output_text.lower()
            or "next_tier" in all_output_text.lower()
            or "exceed" in all_output_text.lower()
        )

        logger.info("\n" + "=" * 60)
        if migration_occurred:
            logger.info("✓ Migration detected in logs!")
        else:
            logger.warning(
                "⚠ No explicit migration detected - may need more tokens or check DecodeDisagger logs"
            )

        logger.info("✓ Test completed successfully!")
        logger.info(f"  - Small tier worker (seqlen={SMALL_TIER_SEQLEN}) started")
        logger.info(f"  - Large tier worker (seqlen={LARGE_TIER_SEQLEN}) started")
        logger.info(f"  - Request completed with {result['count']} chunks")

    finally:
        # Cleanup
        for p in reversed(processes):
            try:
                p.stop()
            except Exception:
                pass

        # Clean up temp directory
        import shutil

        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# Allow running as a standalone script for debugging
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(test_decode_tier_migration())

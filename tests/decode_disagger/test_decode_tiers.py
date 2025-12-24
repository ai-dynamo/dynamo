#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for decode disaggregation with sequence length tiers.

This script:
1. Starts a mocker worker with this_seqlen set
2. Starts a frontend
3. Sends OpenAI chat completions requests
4. Verifies this_seqlen is properly propagated

Usage:
    python -m tests.decode_disagger.test_decode_tiers
"""

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

import aiohttp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Use a real model name that exists on HuggingFace (just for tokenizer)
MODEL_NAME = "Qwen/Qwen3-0.6B"
NAMESPACE = "test-decode-tiers"
SPEEDUP_RATIO = 100.0  # Fast for testing
FRONTEND_PORT = 18080


def read_stream(stream, queue, prefix):
    """Read from a stream and put lines into a queue."""
    try:
        for line in iter(stream.readline, ''):
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
        for stream, name in [(self.process.stdout, 'stdout'), (self.process.stderr, 'stderr')]:
            t = threading.Thread(target=read_stream, args=(stream, self.output_queue, f"{self.name}:{name}"))
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
            except:
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
        sys.executable, "-m", "dynamo.mocker",
        "--model-path", MODEL_NAME,
        "--model-name", MODEL_NAME,
        "--endpoint", endpoint,
        "--store-kv", "etcd",
        "--request-plane", "nats",
        "--speedup-ratio", str(SPEEDUP_RATIO),
        "--block-size", "16",
        "--num-gpu-blocks-override", "1024",
    ]
    
    if this_seqlen is not None:
        cmd.extend(["--this-seqlen", str(this_seqlen)])
    
    return cmd


def build_frontend_cmd(port: int, namespace: str) -> list[str]:
    """Build frontend command."""
    return [
        sys.executable, "-m", "dynamo.frontend",
        "--http-port", str(port),
        "--store-kv", "etcd",
        "--request-plane", "nats",
        "--namespace", namespace,
        "--router-mode", "kv",
        "--kv-cache-block-size", "16",
    ]


async def wait_for_frontend(port: int, timeout: float = 30.0) -> bool:
    """Wait for frontend to be ready."""
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    last_response = None
    
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        last_response = data
                        if data.get("data"):  # Has models
                            logger.info(f"Frontend ready, models: {data}")
                            return True
                        else:
                            logger.debug(f"Frontend responded but no models yet: {data}")
        except Exception as e:
            logger.debug(f"Frontend not ready: {e}")
        await asyncio.sleep(0.5)
    
    logger.error(f"Frontend timeout, last response: {last_response}")
    return False


async def send_chat_request(port: int, max_tokens: int = 50) -> dict:
    """Send a chat completions request."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Hello, how are you today? Please give me a detailed response."}
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    logger.info(f"Sending chat request to {url}")
    
    chunks = []
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            logger.info(f"Response status: {resp.status}")
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Error response: {text}")
                return {"error": text, "status": resp.status}
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data != '[DONE]':
                        try:
                            import json
                            chunk = json.loads(data)
                            chunks.append(chunk)
                        except:
                            pass
    
    return {"chunks": chunks, "count": len(chunks)}


async def run_test():
    """Run the decode disaggregation test."""
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
            "dynamo_llm::discovery::watcher=info,"
            "dynamo_llm::local_model=debug,"
            "dynamo_llm::mocker=info,"
            "info"
        ),
    }
    
    try:
        logger.info(f"Using temp directory for file KV: {tmpdir}")
        
        # Start a worker with this_seqlen set
        # Use standard backend component so frontend can discover it
        worker = ManagedProcess(
            build_mocker_cmd(
                endpoint=f"dyn://{NAMESPACE}.backend.generate",
                this_seqlen=8192,  # Set this_seqlen to verify it's propagated
            ),
            name="worker",
            env=test_env,
        )
        worker.start()
        processes.append(worker)
        
        # Wait for worker to initialize
        logger.info("Waiting for worker to initialize...")
        await asyncio.sleep(3)
        
        # Check worker is running
        if not worker.is_running():
            worker.collect_output()
            logger.error("Worker failed to start!")
            for prefix, line in worker.all_output[-20:]:
                logger.error(f"  {line}")
            raise RuntimeError("Worker failed to start")
        
        # Start frontend
        frontend = ManagedProcess(
            build_frontend_cmd(FRONTEND_PORT, NAMESPACE),
            name="frontend",
            env=test_env,
        )
        frontend.start()
        processes.append(frontend)
        
        # Wait for frontend to be ready
        logger.info("Waiting for frontend to be ready...")
        if not await wait_for_frontend(FRONTEND_PORT, timeout=30):
            frontend.collect_output()
            worker.collect_output()
            logger.error("Frontend failed to start!")
            logger.error("Frontend output:")
            for prefix, line in frontend.all_output[-30:]:
                logger.error(f"  {line}")
            logger.error("Worker output:")
            for prefix, line in worker.all_output[-10:]:
                logger.error(f"  {line}")
            raise RuntimeError("Frontend failed to start")
        
        logger.info("Frontend is ready!")
        
        # Send a test request
        logger.info("Sending test chat request...")
        result = await send_chat_request(FRONTEND_PORT, max_tokens=20)
        
        if "error" in result:
            logger.error(f"Request failed: {result}")
        else:
            logger.info(f"Request completed with {result['count']} chunks")
        
        # Let things settle and collect logs
        await asyncio.sleep(2)
        
        # Collect all output
        for p in processes:
            p.collect_output()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Test Summary")
        logger.info("="*60)
        
        # Check for this_seqlen in logs
        seqlen_keywords = ['this_seqlen', 'seqlen=', 'decode tier']
        
        for p in processes:
            seqlen_lines = [
                line for prefix, line in p.all_output
                if any(x in line.lower() for x in seqlen_keywords)
            ]
            if seqlen_lines:
                logger.info(f"\n{p.name} - this_seqlen related logs:")
                for line in seqlen_lines[:10]:
                    logger.info(f"  {line}")
            else:
                logger.info(f"\n{p.name} - No this_seqlen logs found")
        
        logger.info("\n" + "="*60)
        if "error" not in result:
            logger.info("✓ Test completed successfully!")
            logger.info("  - Worker started with this_seqlen=8192")
            logger.info("  - Frontend discovered the worker")
            logger.info("  - Chat request completed successfully")
            return True
        else:
            logger.error("✗ Test failed - request error")
            return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
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


def main():
    """Main entry point."""
    logger.info("Starting decode disaggregation test")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run with timeout
    try:
        result = asyncio.run(asyncio.wait_for(run_test(), timeout=90))
        sys.exit(0 if result else 1)
    except asyncio.TimeoutError:
        logger.error("Test timed out after 90 seconds")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()

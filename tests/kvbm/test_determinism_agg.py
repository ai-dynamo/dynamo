#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in aggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be 100% match between the two cases. Compared to
disaggregated mode, aggregated mode has less randomness chances.
"""

import importlib.util
import logging
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import pytest
import requests

from .common import DeterminismTester, ServerType
from .common import TestDeterminism as BaseTestDeterminism

# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
]


class LLMServerManager:
    """Manages LLM server lifecycle for determinism testing."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        self.server_type = server_type
        self.port = port or int(os.environ.get("KVBM_SERVER_PORT", "8000"))
        self.base_url = base_url or f"http://localhost:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks

        # Prepare logging
        self.log_dir = log_dir or Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = (
            f"cpu{cpu_cache_blocks or 'default'}_gpu{gpu_cache_blocks or 'default'}"
        )
        self.server_log_file = (
            self.log_dir / f"{self.server_type}_server_{config_str}_{timestamp}.log"
        )
        self.server_stdout_file: Optional[TextIO] = None
        self.server_stderr_file: Optional[TextIO] = None

        # Environment for the process
        self.env = os.environ.copy()
        self.env.update(
            {
                "RUST_BACKTRACE": "1",
                # DynamoConnector connection settings
                "NATS_SERVER": "nats://localhost:4222",
                "ETCD_ENDPOINTS": "http://localhost:2379",
            }
        )

        # CPU cache blocks override via env
        if cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config(gpu_cache_blocks)
        elif self.server_type == ServerType.trtllm:
            self._set_up_trtllm_config(gpu_cache_blocks)
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in the KVBM test suite"
            )

    def _set_up_vllm_config(self, gpu_cache_blocks):
        self.env["VLLM_SERVER_DEV_MODE"] = "1"

        # Construct serve command
        self.server_cmd = [
            "vllm",
            "serve",
            "--block-size",
            "16",
            "--port",
            str(self.port),
            "--kv-transfer-config",
            '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "dynamo.llm.vllm_integration.connector"}',
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--max-seq-len",
            "8000",  # required to fit on L4 GPU when using 8b model
        ]

        # GPU blocks override
        if gpu_cache_blocks is not None:
            self.server_cmd.extend(["--num-gpu-blocks-override", str(gpu_cache_blocks)])

    def _set_up_trtllm_config(self, gpu_cache_blocks):
        config_path = os.environ.get(
            "KVBM_TRTLLM_LLMAPI_CONFIG_PATH", "/tmp/kvbm_llm_api_config.yaml"
        )
        llm_api_config: Dict[str, Any] = {}
        llm_api_config[
            "cuda_graph_config"
        ] = None  # explicitly disable CUDA graph since Connector API doesn't support CUDA graph yet in TRTLLM
        llm_api_config["kv_cache_config"] = {
            "enable_partial_reuse": False,
            "free_gpu_memory_fraction": 0.10,  # Set a small GPU fraction so that we can evict/reset the on-device kv cache faster
        }
        llm_api_config["kv_connector_config"] = {
            "connector_module": "dynamo.llm.trtllm_integration.connector",
            "connector_scheduler_class": "DynamoKVBMConnectorLeader",
            "connector_worker_class": "DynamoKVBMConnectorWorker",
        }

        # GPU blocks override
        if gpu_cache_blocks is not None:
            del llm_api_config["kv_cache_config"]["free_gpu_memory_fraction"]
            llm_api_config["kv_cache_config"]["max_tokens"] = (
                int(gpu_cache_blocks) * 32
            )  # TRTLLM defaults 32 tokens per block

        # Construct serve command
        self.server_cmd = [
            "trtllm-serve",
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--host",
            "localhost",
            "--port",
            str(self.port),
            "--backend",
            "pytorch",
            "--extra_llm_api_options",
            config_path,
        ]

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(llm_api_config, f, default_flow_style=False, sort_keys=False)

    def start_server(self, timeout: int = 300) -> bool:
        """Start LLM server and wait for readiness."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(2)

        # Open log files
        self.server_stdout_file = open(
            self.server_log_file.with_suffix(".stdout.log"), "w"
        )
        self.server_stderr_file = open(
            self.server_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.server_stdout_file is not None:
            self.server_stdout_file.write(
                f"=== {self.server_type} Server Started at {datetime.now()} ===\nCommand: {' '.join(self.server_cmd)}\n"
            )
            self.server_stdout_file.flush()

        # Launch
        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=self.server_stdout_file,
            stderr=self.server_stderr_file,
            env=self.env,
            preexec_fn=os.setsid,
        )

        # Wait for health
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                return True
            if self.process.poll() is not None:
                self._close_log_files()
                return False
            time.sleep(5)

        # Timeout
        self.stop_server()
        return False

    def stop_server(self):
        """Stop LLM server and close logs."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process = None
        self._close_log_files()

    def _close_log_files(self):
        if self.server_stdout_file:
            self.server_stdout_file.write(
                f"\n=== Server Stopped at {datetime.now()} ===\n"
            )
            self.server_stdout_file.close()
            self.server_stdout_file = None
        if self.server_stderr_file:
            self.server_stderr_file.close()
            self.server_stderr_file = None

    def is_server_running(self) -> bool:
        try:
            # First check basic health
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                return False

            # Then check if the model endpoint is ready with a simple test request
            test_payload = {
                "model": os.environ.get(
                    "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                ),
                "messages": [{"role": "user", "content": "test"}],
                "max_completion_tokens": 1,
                "temperature": 0,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=10,
            )
            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False


class AggDeterminismTester(DeterminismTester):
    """Aggregated architecture specific determinism tester."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        print("Resetting prefix cache...")
        if self.server_type == ServerType.trtllm:
            # TRTLLM doesn't support reset_prefix_cache endpoint API
            # 300 shakespeare content could evict the 0.1 x 80G (~1700 blocks) on-device cache
            shakespeare_count = 300
            for seq_idx in range(1, shakespeare_count + 1):
                start_word = (seq_idx - 1) * self.word_count
                content = self.get_shakespeare_content(start_word)

                if content:
                    print(
                        f"Resetting Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    try:
                        self.make_request(content)
                    except Exception as e:
                        print(f"Resetting request failed: {e}")
        else:
            response = requests.post(
                f"{self.base_url}/reset_prefix_cache",
                timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
            )
            response.raise_for_status()
        print("Cache reset done")


@pytest.fixture(scope="function")
def llm_server(request, runtime_services):
    """Start and stop a LLM server for each test with optional cache block overrides.

    To parametrize, use:
      @pytest.mark.parametrize("llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 2048}], indirect=True)
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)

    # Put logs in the per-test directory set up by tests/conftest.py
    log_dir = Path(request.node.name)

    if importlib.util.find_spec("vllm") is not None:
        server_type = ServerType.vllm
    elif importlib.util.find_spec("tensorrt_llm") is not None:
        server_type = ServerType.trtllm
    else:
        raise Exception(
            "Neither the vllm nor the tensorrt_llm module is available in the current environment."
        )

    server_manager = LLMServerManager(
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        log_dir=log_dir,
        server_type=server_type,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start {server_type} server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def tester(llm_server):
    """Create determinism tester bound to the running server's base URL."""
    t = AggDeterminismTester(
        base_url=llm_server.base_url, server_type=llm_server.server_type
    )
    t.download_shakespeare_text()
    return t


class TestDeterminismAgg(BaseTestDeterminism):
    """Test class for determinism validation."""

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    def test_determinism_agg_with_cache_reset(
        self, tester, llm_server, runtime_services
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        # Call the base class implementation
        super().base_test_determinism_with_cache_reset(
            tester, llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "20000"))},
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "num_concurrent",
        [int(x) for x in os.environ.get("KVBM_CONCURRENT_REQUESTS", "3").split(",")],
    )
    @pytest.mark.parametrize(
        "max_tokens",
        [int(x) for x in os.environ.get("KVBM_MAX_TOKENS", "10").split(",")],
    )
    @pytest.mark.parametrize(
        "num_prompts",
        [int(x) for x in os.environ.get("KVBM_IFEVAL_PROMPTS", "120").split(",")],
    )
    @pytest.mark.skip(reason="Flaky test: DIS-665")
    def test_concurrent_determinism_with_ifeval(
        self,
        tester,
        llm_server,
        runtime_services,
        num_concurrent,
        max_tokens,
        num_prompts,
    ):
        """Simple concurrent determinism test: send IFEval prompts concurrently, with cache reset."""
        print("\n" + "=" * 70)
        print("CONCURRENT DETERMINISM TEST WITH IFEVAL")
        print("=" * 70)

        # Override max_tokens for this test iteration
        original_max_tokens = os.environ.get("KVBM_MAX_TOKENS")
        os.environ["KVBM_MAX_TOKENS"] = str(max_tokens)
        print(
            f"Using KVBM_MAX_TOKENS={max_tokens} (parametrized, original: {original_max_tokens or '48'})"
        )

        # Configuration comes from parametrize
        print(
            f"Configuration: {num_concurrent} concurrent requests, {max_tokens} max tokens"
        )

        # Load IFEval prompts
        ifeval_prompts = tester.download_ifeval_dataset()
        if not ifeval_prompts:
            pytest.skip("IFEval dataset not available")

        # Use parametrized number of IFEval prompts
        test_prompts = ifeval_prompts[:num_prompts]
        print(
            f"Using {len(test_prompts)} IFEval prompts for concurrent testing (parametrized: {num_prompts})"
        )
        print(f"Concurrency level: {num_concurrent} simultaneous requests")

        # Show sample prompts
        print("\nSample prompts:")
        for i, prompt in enumerate(test_prompts[:3]):
            print(f"  {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        if len(test_prompts) > 3:
            print(f"  ... and {len(test_prompts) - 3} more")

        def run_concurrent_test(phase_name, do_warmup=False):
            """Run one phase of concurrent testing."""
            print(f"\n=== {phase_name} ===")

            if do_warmup:
                # KV Cache warmup - send ALL test prompts to compute KV caches
                print(
                    f"Warming up KV caches with all {len(test_prompts)} test prompts..."
                )
                warmup_failed = 0

                for i, prompt in enumerate(test_prompts):
                    if (
                        i % 5 == 0 or i == len(test_prompts) - 1
                    ):  # Progress every 5 prompts
                        print(f"  Warmup progress: {i+1}/{len(test_prompts)}")

                    try:
                        tester.make_request(prompt)
                    except Exception as e:
                        warmup_failed += 1
                        if warmup_failed <= 3:  # Show first few failures
                            print(f"    Warmup failed for prompt {i}: {e}")

                if warmup_failed > 0:
                    print(
                        f"Warmup completed with {warmup_failed} failures out of {len(test_prompts)} prompts"
                    )
                else:
                    print(
                        f"Warmup completed successfully - all {len(test_prompts)} KV caches computed"
                    )

                # Wait for 10 seconds to make sure all transfers are complete
                time.sleep(10)
            else:
                print("Skipping warmup (already done in previous phase)")

            # Run concurrent requests
            print(
                f"Sending {len(test_prompts)} requests with {num_concurrent} max concurrent..."
            )
            start_time = time.time()

            def make_request_wrapper(prompt_and_idx):
                idx, prompt = prompt_and_idx
                try:
                    response = tester.make_request(prompt)
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "response": response,
                        "success": True,
                    }
                except Exception as e:
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                    }

            # Execute all requests concurrently
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                results = list(
                    executor.map(make_request_wrapper, enumerate(test_prompts))
                )

            elapsed = time.time() - start_time
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            print(
                f"Completed in {elapsed:.2f}s - Success: {len(successful)}, Failed: {len(failed)}"
            )

            if failed:
                for fail in failed[:3]:  # Show first few failures
                    print(f"  Failed: {fail['error']}")

            return successful

        # Phase 1: Before cache reset
        results_before = run_concurrent_test(
            "PHASE 1: BEFORE CACHE RESET", do_warmup=True
        )

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        tester.reset_prefix_cache()

        # Phase 2: After cache reset
        results_after = run_concurrent_test("PHASE 2: AFTER CACHE RESET")

        # Compare results between phases
        print("\n" + "=" * 70)
        print("DETERMINISM ANALYSIS")
        print("=" * 70)

        # Create lookup for before results
        before_responses = {r["idx"]: r["response"] for r in results_before}
        after_responses = {r["idx"]: r["response"] for r in results_after}

        deterministic_count = 0
        total_compared = 0

        for idx in before_responses:
            if idx in after_responses:
                total_compared += 1
                before_resp = before_responses[idx]
                after_resp = after_responses[idx]

                if before_resp == after_resp:
                    deterministic_count += 1
                    print(f"   Prompt {idx}: DETERMINISTIC")
                else:
                    print(f"   Prompt {idx}: NON-DETERMINISTIC")
                    print(f"     Before: {before_resp}")
                    print(f"     After:  {after_resp}")

        # Final assessment
        success_rate = deterministic_count / total_compared if total_compared > 0 else 0
        print("\n=== FINAL RESULT ===")
        print(f"Prompts compared: {total_compared}")
        print(f"Deterministic: {deterministic_count}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Concurrent requests: {num_concurrent}")

        # Restore original max_tokens setting
        if original_max_tokens is not None:
            os.environ["KVBM_MAX_TOKENS"] = original_max_tokens
        else:
            os.environ.pop("KVBM_MAX_TOKENS", None)

        assert (
            success_rate == 1.0
        ), f"Determinism failed: {deterministic_count}/{total_compared} prompts deterministic"


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for language model API using pytest.

This test suite checks if the model produces deterministic outputs
when given the same inputs with fixed seed and temperature=0.

The test uses comprehensive server warmup (sending all test prompts
before validation) to avoid server initialization effects that could
impact determinism measurements.
"""

import importlib.util
import logging
import os
import signal
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import pytest
import requests

# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci
pytestmark = [
    pytest.mark.kvbm,
    #    pytest.mark.e2e,
    #    pytest.mark.slow,
    #    pytest.mark.nightly,
    #    pytest.mark.gpu_1,
]


class ServerType(str, Enum):
    vllm = "vllm"
    trtllm = "trtllm"


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
                "DYN_LOG": os.environ.get(
                    "DYN_LOG", "debug,dynamo_llm::block_manager::layout=error"
                ),
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
        ]

        # GPU blocks override
        if gpu_cache_blocks is not None:
            self.server_cmd.extend(["--num-gpu-blocks-override", str(gpu_cache_blocks)])

    def _set_up_trtllm_config(self, gpu_cache_blocks):
        config_path = os.environ.get(
            "KVBM_TRTLLM_LLMAPI_CONFIG_PATH", "/tmp/kvbm_llm_api_config.yaml"
        )
        llm_api_config: dict[str, Any] = {}
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


class DeterminismTester:
    """Test class for model determinism validation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        # Allow environment override for flexibility in CI/local runs
        self.base_url = (
            base_url or os.environ.get("DYNAMO_API_BASE_URL") or "http://localhost:8000"
        )
        self.model_id = (
            model_id
            or os.environ.get("KVBM_MODEL_ID")
            or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        self.server_type = server_type

        self.shakespeare_file = Path("t8.shakespeare.txt")
        self.max_iterations = int(os.environ.get("KVBM_MAX_ITERATIONS", "500"))
        self.word_count = int(os.environ.get("KVBM_WORD_COUNT", "200"))

        # Test intervals
        self.control_interval = int(os.environ.get("KVBM_CONTROL_INTERVAL", "10"))
        self.shakespeare_interval = int(
            os.environ.get("KVBM_SHAKESPEARE_INTERVAL", "1")
        )
        self.random_interval = int(os.environ.get("KVBM_RANDOM_INTERVAL", "7"))

        # Response storage
        self.control_responses: Dict[int, List[str]] = defaultdict(list)
        self.shakespeare_responses: Dict[int, List[str]] = defaultdict(list)
        self.random_responses: Dict[int, List[str]] = defaultdict(list)

        # Control sequences
        self.control_sequences = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog. This is a standard pangram that contains all letters of the alphabet.",
            "Find light in the beautiful sea, I choose to be happy, You and I, you and I, we are like a beautiful melody that never ends, dancing through the night with stars as our companions, whispering secrets to the wind as we journey through life together, hand in hand, heart to heart, forever and always.",
            "The advancement of technology has fundamentally transformed the way we live, work, and communicate in the modern world. From the invention of the printing press to the development of the internet, each technological breakthrough has opened new possibilities and created unprecedented opportunities for human progress. Today, artificial intelligence and machine learning are reshaping industries, healthcare, education, and countless other fields, promising to solve complex problems and improve the quality of life for people around the globe.",
            "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
            "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons, each connected to thousands of others through intricate networks of synapses. This biological supercomputer processes information at speeds that would make even the most advanced artificial intelligence systems seem primitive by comparison. Every thought, memory, emotion, and decision we make is the result of electrical and chemical signals traveling through this vast neural network. The brain's ability to learn, adapt, and create is unmatched by any machine we have ever built. It can recognize patterns in milliseconds, solve complex problems through intuition, and generate creative ideas that have never existed before. Yet despite our incredible advances in neuroscience, we still understand only a fraction of how this remarkable organ truly works. The mysteries of consciousness, memory formation, and the nature of human intelligence continue to challenge the brightest minds in science and philosophy.",
        ]

        # Random sequences
        self.random_sequences = [
            "Coffee is ready",
            "The cat sat on the mat while the dog slept peacefully in the corner, creating a perfect picture of domestic tranquility that warmed the heart of anyone who witnessed this simple moment of harmony between two natural enemies turned friends.",
            "Mathematics is the language of the universe, and numbers are its alphabet. Through the elegant dance of equations and the symphony of algorithms, we unlock the secrets of nature's most profound mysteries. From the simple beauty of prime numbers to the complex elegance of calculus, mathematics provides us with the tools to understand everything from the smallest subatomic particles to the vast expanse of galaxies stretching across the cosmic void.",
            "A journey of a thousand miles begins with a single step, as the ancient Chinese proverb wisely reminds us. This timeless wisdom speaks to the fundamental truth that every great achievement, every monumental discovery, and every life-changing transformation starts with that crucial moment of decision - the moment when we choose to take action instead of remaining in the comfort of inaction. Whether it's learning a new skill, starting a business, writing a novel, or embarking on a spiritual quest, the path to success is paved with countless small steps, each one building upon the last, until we find ourselves transformed by the journey itself.",
            "Technology evolves rapidly, but human nature remains constant through the ages. Despite the incredible advances in artificial intelligence, virtual reality, and biotechnology, the fundamental desires, fears, and aspirations that drive human behavior have remained remarkably consistent throughout history. We still seek connection, meaning, and purpose in our lives. We still fear the unknown and crave security. We still dream of a better future and work to create it for ourselves and our loved ones. This paradox - the ever-changing nature of our tools and the unchanging nature of our hearts - is perhaps the most fascinating aspect of the human condition, reminding us that while we may build increasingly sophisticated machines, we remain fundamentally human in our core essence.",
        ]

    def download_shakespeare_text(self):
        """Download Shakespeare text if not present."""
        if not self.shakespeare_file.exists():
            print("Downloading Shakespeare text...")
            import urllib.request

            url = os.environ.get(
                "KVBM_SHAKESPEARE_URL",
                "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt",
            )
            urllib.request.urlretrieve(url, self.shakespeare_file)

            # Remove double newlines
            with open(self.shakespeare_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("\n\n", "")
            with open(self.shakespeare_file, "w", encoding="utf-8") as f:
                f.write(content)

    def make_request(self, content: str) -> str:
        """Make API request and return completion text."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": content},
            ],
            "stream": False,
            "max_completion_tokens": int(os.environ.get("KVBM_MAX_TOKENS", "48")),
            "temperature": 0,
            "top_p": 0.0001,
            "seed": int(os.environ.get("KVBM_SEED", "42")),
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

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

    def warmup_server(self):
        """Perform comprehensive server warmup with all test prompts."""
        print("=" * 70)
        print("PERFORMING COMPREHENSIVE SERVER WARMUP")
        print("=" * 70)
        print(
            "Sending all control, Shakespeare, and random prompts to warm up the server..."
        )

        # Warmup with all control sequences
        print("Warming up with control sequences...")
        for i, control_seq in enumerate(self.control_sequences):
            print(f"  Warmup control sequence {i + 1}: {control_seq[:50]}...")
            try:
                self.make_request(control_seq)
            except Exception as e:
                print(f"  Warning: Warmup request failed: {e}")

        # Warmup with Shakespeare sequences that will be used in testing
        print("Warming up with Shakespeare sequences...")
        shakespeare_count = self.max_iterations // self.shakespeare_interval
        for seq_idx in range(1, shakespeare_count + 1):
            start_word = (seq_idx - 1) * self.word_count
            content = self.get_shakespeare_content(start_word)

            if content:
                print(
                    f"  Warmup Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                )
                try:
                    self.make_request(content)
                except Exception as e:
                    print(f"  Warning: Warmup request failed: {e}")

        # Warmup with all random sequences
        print("Warming up with random sequences...")
        for i, random_seq in enumerate(self.random_sequences):
            print(f"  Warmup random sequence {i + 1}: {random_seq[:50]}...")
            try:
                self.make_request(random_seq)
            except Exception as e:
                print(f"  Warning: Warmup request failed: {e}")

        print("Server warmup completed!")
        print("=" * 70)

    def get_shakespeare_content(self, start_word: int) -> str:
        """Get Shakespeare content starting from a specific word."""
        with open(self.shakespeare_file, "r", encoding="utf-8") as f:
            words = f.read().split()

        end_word = min(start_word + self.word_count, len(words))
        return " ".join(words[start_word:end_word])

    def download_ifeval_dataset(self) -> List[str]:
        """Download and extract all prompts from IFEval dataset."""
        try:
            from datasets import load_dataset

            print("Loading complete IFEval dataset...")
            dataset = load_dataset("google/IFEval", split="train")

            # Extract all prompts from the dataset
            prompts = []

            for example in dataset:
                # IFEval has 'prompt' field with the instruction
                if "prompt" in example:
                    prompt_text = example["prompt"].strip()
                    if prompt_text:  # Only skip empty prompts
                        prompts.append(prompt_text)

            print(f"Loaded {len(prompts)} prompts from complete IFEval dataset")
            return prompts

        except ImportError:
            print(
                "Warning: datasets library not available, falling back to default prompts"
            )
            return self.control_sequences + self.random_sequences
        except Exception as e:
            print(
                f"Warning: Failed to load IFEval dataset ({e}), falling back to default prompts"
            )
            return self.control_sequences + self.random_sequences

    def run_test_iterations(self):
        """Run the test iterations with comprehensive warmup."""
        # Perform initial warmup before testing
        self.warmup_server()

        for iteration in range(1, self.max_iterations + 1):
            print(f"Iteration {iteration}/{self.max_iterations}")

            # Control sequence test
            if iteration % self.control_interval == 0:
                control_idx = (iteration // self.control_interval - 1) % len(
                    self.control_sequences
                )
                control_content = self.control_sequences[control_idx]

                print(
                    f"  Running control sequence {control_idx + 1}: {control_content[:50]}..."
                )
                completion = self.make_request(control_content)
                self.control_responses[control_idx].append(completion)
                print(f"  Response: {completion}")

            # Shakespeare sequence test
            if iteration % self.shakespeare_interval == 0:
                start_word = (
                    iteration // self.shakespeare_interval - 1
                ) * self.word_count
                content = self.get_shakespeare_content(start_word)

                if content:
                    shakespeare_idx = iteration // self.shakespeare_interval - 1
                    print(
                        f"  Running Shakespeare sequence {shakespeare_idx + 1} (words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    completion = self.make_request(content)
                    self.shakespeare_responses[shakespeare_idx].append(completion)
                    print(f"  Response: {completion}")

            # Random sequence test
            if iteration % self.random_interval == 0:
                random_idx = (iteration // self.random_interval - 1) % len(
                    self.random_sequences
                )
                random_content = self.random_sequences[random_idx]

                print(
                    f"  Running random sequence {random_idx + 1}: {random_content[:50]}..."
                )
                completion = self.make_request(random_content)
                self.random_responses[random_idx].append(completion)
                print(f"  Response: {completion}")

    def analyze_responses(
        self, responses: Dict[int, List[str]], sequence_type: str
    ) -> Tuple[int, int]:
        """Analyze responses for determinism."""
        passed = 0
        failed = 0

        print(f"\n=== {sequence_type.upper()} SEQUENCES ===")

        for idx, response_list in responses.items():
            if not response_list:
                continue

            print(f"\n{sequence_type} sequence {idx + 1}:")
            print(f"Total responses: {len(response_list)}")

            if len(response_list) == 1:
                print("Single response - cannot check determinism")
                continue

            reference = response_list[0]
            differences = 0

            print(f"Reference response: {reference}")

            for i, response in enumerate(response_list[1:], 2):
                if response == reference:
                    print(f"Response {i}: MATCHES reference")
                else:
                    print(f"Response {i}: DIFFERS from reference")
                    print(f"  Expected: {reference}")
                    print(f"  Got:      {response}")
                    differences += 1

            if differences == 0:
                print(" ALL RESPONSES IDENTICAL - DETERMINISTIC")
                passed += 1
            else:
                print(f" {differences} DIFFERENCES DETECTED - NON-DETERMINISTIC")
                failed += 1

        return passed, failed

    def test_concurrent_determinism(
        self, prompts: List[str], num_workers: int = 4, requests_per_prompt: int = 3
    ) -> bool:
        """Test determinism with concurrent requests to the same prompts."""
        print("\n=== CONCURRENT DETERMINISM TEST ===")
        print(f"Workers: {num_workers}, Requests per prompt: {requests_per_prompt}")

        # Prepare test data: each prompt will get multiple concurrent requests
        test_tasks = []
        for i, prompt in enumerate(prompts):
            for req_num in range(requests_per_prompt):
                test_tasks.append(
                    {
                        "prompt_idx": i,
                        "prompt": prompt,
                        "request_id": f"p{i}_r{req_num}",
                    }
                )

        print(f"Total concurrent requests: {len(test_tasks)}")

        # Storage for responses grouped by prompt
        concurrent_responses: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

        def make_concurrent_request(task):
            """Worker function for concurrent requests."""
            try:
                response = self.make_request(task["prompt"])
                return {
                    "prompt_idx": task["prompt_idx"],
                    "request_id": task["request_id"],
                    "response": response,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "prompt_idx": task["prompt_idx"],
                    "request_id": task["request_id"],
                    "response": None,
                    "success": False,
                    "error": str(e),
                }

        # Execute concurrent requests
        print("Executing concurrent requests...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(make_concurrent_request, task): task
                for task in test_tasks
            }

            # Collect results
            completed = 0
            failed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                completed += 1

                if result["success"]:
                    concurrent_responses[result["prompt_idx"]].append(
                        (result["request_id"], result["response"])
                    )
                    if completed % 10 == 0:
                        print(f"  Completed: {completed}/{len(test_tasks)}")
                else:
                    failed += 1
                    print(f"  Failed request {result['request_id']}: {result['error']}")

        elapsed = time.time() - start_time
        print(
            f"Completed {completed} requests in {elapsed:.2f}s ({completed/elapsed:.1f} req/s)"
        )
        print(f"Failed requests: {failed}")

        # Analyze concurrent determinism
        print("\n=== CONCURRENT DETERMINISM ANALYSIS ===")
        total_prompts_tested = 0
        deterministic_prompts = 0

        for prompt_idx, responses in concurrent_responses.items():
            if len(responses) < 2:
                print(
                    f"Prompt {prompt_idx}: Only {len(responses)} response(s), skipping"
                )
                continue

            total_prompts_tested += 1
            prompt_text = prompts[prompt_idx]
            print(f"\nPrompt {prompt_idx}: {prompt_text[:50]}...")
            print(f"Concurrent responses: {len(responses)}")

            # Extract just the response text
            response_texts = [resp[1] for resp in responses]
            request_ids = [resp[0] for resp in responses]

            # Check if all responses are identical
            reference_response = response_texts[0]
            mismatches = []

            for req_id, response_text in zip(request_ids[1:], response_texts[1:]):
                if response_text != reference_response:
                    mismatches.append((req_id, response_text))

            if not mismatches:
                print(
                    f"   DETERMINISTIC: All {len(responses)} concurrent responses identical"
                )
                print(f"     Response: {reference_response}")
                deterministic_prompts += 1
            else:
                print(f"    NON-DETERMINISTIC: {len(mismatches)} different responses")
                print(f"    Reference ({request_ids[0]}): {reference_response}")
                for req_id, diff_response in mismatches:
                    print(f"     Different ({req_id}): {diff_response}")

        # Final assessment
        success_rate = (
            deterministic_prompts / total_prompts_tested
            if total_prompts_tested > 0
            else 0
        )
        print("\n=== FINAL CONCURRENT DETERMINISM RESULT ===")
        print(f"Prompts tested: {total_prompts_tested}")
        print(f"Deterministic: {deterministic_prompts}")
        print(f"Non-deterministic: {total_prompts_tested - deterministic_prompts}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Concurrency level: {num_workers} workers")
        print(f"Request rate: {completed/elapsed:.1f} req/s")

        return success_rate == 1.0


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
    t = DeterminismTester(
        base_url=llm_server.base_url, server_type=llm_server.server_type
    )
    t.download_shakespeare_text()
    return t


class TestDeterminism:
    """Test class for determinism validation."""

    @pytest.mark.parametrize(
        "llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    def test_determinism_with_cache_reset(self, tester, llm_server, runtime_services):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        print("\n" + "=" * 70)
        print("STARTING DETERMINISM TEST (WITH CACHE RESET)")
        print("=" * 70)

        # Phase 1: Run test with warmup
        print("\n=== PHASE 1: BEFORE CACHE RESET (WITH WARMUP) ===")
        tester.run_test_iterations()

        # Store Phase 1 results
        phase1_control = {k: v.copy() for k, v in tester.control_responses.items()}
        phase1_shakespeare = {
            k: v.copy() for k, v in tester.shakespeare_responses.items()
        }
        phase1_random = {k: v.copy() for k, v in tester.random_responses.items()}

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        tester.reset_prefix_cache()

        # Clear response storage for Phase 2 (they are defaultdict, so they'll auto-initialize)
        tester.control_responses.clear()
        tester.shakespeare_responses.clear()
        tester.random_responses.clear()

        # Phase 2: Run test without warmup
        print("\n=== PHASE 2: AFTER CACHE RESET (NO WARMUP) ===")
        # Temporarily disable warmup by modifying the method
        original_warmup = tester.warmup_server
        tester.warmup_server = lambda: print(
            "Skipping warmup (testing determinism across cache reset)"
        )

        try:
            tester.run_test_iterations()
        finally:
            # Restore original warmup method
            tester.warmup_server = original_warmup

        # Compare Phase 1 vs Phase 2 results
        print("\n" + "=" * 70)
        print("CROSS-CACHE-RESET DETERMINISM ANALYSIS")
        print("=" * 70)

        total_passed = 0
        total_failed = 0

        # Compare control sequences
        for seq_idx in phase1_control:
            if seq_idx in tester.control_responses:
                phase1_responses = phase1_control[seq_idx]
                phase2_responses = tester.control_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Control {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(f"   Control {seq_idx}, response {i}: NON-DETERMINISTIC")
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Compare Shakespeare sequences
        for seq_idx in phase1_shakespeare:
            if seq_idx in tester.shakespeare_responses:
                phase1_responses = phase1_shakespeare[seq_idx]
                phase2_responses = tester.shakespeare_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Shakespeare {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(
                            f"   Shakespeare {seq_idx}, response {i}: NON-DETERMINISTIC"
                        )
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Compare random sequences
        for seq_idx in phase1_random:
            if seq_idx in tester.random_responses:
                phase1_responses = phase1_random[seq_idx]
                phase2_responses = tester.random_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Random {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(f"   Random {seq_idx}, response {i}: NON-DETERMINISTIC")
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Final assessment
        print("\n" + "=" * 70)
        print("FINAL CROSS-CACHE-RESET DETERMINISM ASSESSMENT")
        print("=" * 70)
        print(f"Total comparisons: {total_passed + total_failed}")
        print(f"Passed (deterministic): {total_passed}")
        print(f"Failed (non-deterministic): {total_failed}")
        print(
            "Test compared responses before cache reset (with warmup) vs after cache reset (no warmup)."
        )

        if total_passed + total_failed == 0:
            pytest.skip("No tests were completed - insufficient data")

        assert (
            total_failed == 0
        ), f"Model is not deterministic across cache reset: {total_failed} comparisons failed"

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
                # Reset cache
                print("\n" + "=" * 50)
                print("RESETTING CACHE AFTER WARMUP")
                print("=" * 50)
                tester.reset_prefix_cache()
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

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common functionality for KVBM determinism tests.

This module contains shared classes and functions used by both
aggregated and disaggregated determinism tests.
"""

import importlib.util
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import requests

# ============================================================================
# Module Availability Checks
# ============================================================================


def check_module_available(module_name: str) -> bool:
    """Check if a Python module is available and importable.

    This function first checks if the module spec can be found, then attempts
    to actually import it to ensure it's functional.

    Args:
        module_name: Name of the module to check (e.g., "vllm", "tensorrt_llm")

    Returns:
        True if the module is available and importable, False otherwise

    Example:
        >>> has_vllm = check_module_available("vllm")
        >>> has_trtllm = check_module_available("tensorrt_llm")
    """
    if importlib.util.find_spec(module_name) is None:
        return False
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using character-level matching.

    Returns a similarity ratio between 0 and 1:
    - 1.0 = exact match
    - 0.8+ = semantically equivalent (minor word changes)
    - <0.7 = significantly different
    """
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def are_semantically_equivalent(
    text1: str,
    text2: str,
    min_similarity: float = 0.75,
    prefix_exact_match_ratio: float = 0.5,
) -> tuple:
    """
    Check if two texts are semantically equivalent.

    Checks both overall similarity and prefix matching to ensure early tokens
    are deterministic (where FP errors haven't accumulated).

    Args:
        text1: First text (baseline)
        text2: Second text (response to compare)
        min_similarity: Minimum similarity ratio (0-1) to consider equivalent
        prefix_exact_match_ratio: Ratio of text that must exactly match from start

    Returns:
        (is_equivalent, similarity_score, reason)
    """
    # Calculate overall similarity
    similarity = calculate_semantic_similarity(text1, text2)

    # Check prefix match (first X% must be exact to avoid early divergence)
    prefix_len = int(min(len(text1), len(text2)) * prefix_exact_match_ratio)
    prefix_match = text1[:prefix_len] == text2[:prefix_len]

    if similarity >= min_similarity:
        if prefix_match:
            return (
                True,
                similarity,
                f"Semantically equivalent ({similarity:.1%} similar, prefix matches)",
            )
        else:
            return (
                False,
                similarity,
                f"High similarity but early divergence (prefix mismatch at {prefix_len} chars)",
            )
    else:
        return (False, similarity, f"Low similarity ({similarity:.1%})")


def load_prompt_from_file(prompt_path: Path) -> Optional[str]:
    """Load and preprocess prompt from file.

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Cleaned prompt content, or None if file doesn't exist
    """
    if not prompt_path.exists():
        return None

    with open(prompt_path, "r", encoding="utf-8") as f:
        # Strip SPDX license header lines (start with #)
        lines = f.readlines()
        content_lines = [line for line in lines if not line.startswith("#")]
        return "".join(content_lines).strip()


def check_logs_for_patterns(
    log_path: Path, patterns: List[str], process_name: str
) -> List[str]:
    """Check log file for specific patterns (errors, warnings, etc.)."""
    findings = []

    if not log_path.exists():
        return [f"{process_name} log file not found at {log_path}"]

    try:
        with open(log_path, "r") as f:
            content = f.read()

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Limit to first 3 matches and truncate each to 200 chars
                    for match in matches[:3]:
                        match_str = match if isinstance(match, str) else str(match)
                        findings.append(f"{process_name}: {match_str[:200]}")
    except Exception as e:
        findings.append(f"Error reading {process_name} log: {e}")

    return findings


class ApiTester:
    """Base class for making API requests to LLM endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("DYNAMO_API_BASE_URL") or "http://localhost:8000"
        )
        self.model_id = model_id or os.environ.get(
            "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )

    def make_request(
        self,
        content: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Make API request and return completion text."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": content},
            ],
            "stream": False,
            "temperature": temperature,
            "seed": seed,
        }

        # Add max_tokens with appropriate key based on kwargs or defaults
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        else:
            payload["max_completion_tokens"] = int(
                os.environ.get("KVBM_MAX_TOKENS", "48")
            )

        # Add any additional kwargs
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def send_chat_request(
        self,
        messages: List[dict],
        max_tokens: int = 50,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> dict:
        """Send a chat request and return full response JSON."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()


class ServerType(str, Enum):
    vllm = "vllm"
    trtllm = "trtllm"


class DeterminismTester(ApiTester):
    """Test class for model determinism validation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id)
        self.server_type = server_type

        self.shakespeare_file = Path("t8.shakespeare.txt")
        self.max_iterations = int(os.environ.get("KVBM_MAX_ITERATIONS", "100"))
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

    # Inherited from ApiTester, but override to add top_p for determinism testing
    def make_request(
        self,
        content: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Make API request and return completion text with determinism settings."""
        # Use determinism-specific defaults
        if max_tokens is None:
            max_tokens = int(os.environ.get("KVBM_MAX_TOKENS", "48"))
        if seed == 42:  # Default seed, use env override
            seed = int(os.environ.get("KVBM_SEED", "42"))

        return super().make_request(
            content,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=0.0001,  # For determinism
            **kwargs,
        )

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


@pytest.fixture(scope="function")
def tester(llm_server):
    """Create determinism tester bound to the running server's base URL."""
    t = DeterminismTester(
        base_url=llm_server.base_url, server_type=llm_server.server_type
    )
    t.download_shakespeare_text()
    return t


@pytest.fixture(scope="function")
def llm_server_kvbm(request, runtime_services):
    """Start LLM server with configurable cache sizes for KVBM testing.

    Usage in test files:
        @pytest.mark.parametrize("llm_server_kvbm",
            [{"cpu_blocks": 100, "gpu_blocks": 10, "model": "Qwen/Qwen3-0.6B"}], indirect=True)
        def test_example(llm_server_kvbm):
            ...
    """
    import os
    import time

    from tests.utils.managed_process import ManagedProcess

    # Get configuration from request.param
    params = getattr(request, "param", {})
    cpu_blocks = params.get("cpu_blocks", 100)
    gpu_blocks = params.get("gpu_blocks", 10)
    model = params.get(
        "model",
        os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    )

    # Detect available server type
    if check_module_available("vllm"):
        server_type = ServerType.vllm
    elif check_module_available("tensorrt_llm"):
        server_type = ServerType.trtllm
        pytest.skip("TensorRT-LLM tests are disabled for this test")
    else:
        pytest.skip(
            "Neither vllm nor tensorrt_llm module is available in the current environment."
        )

    # Build vLLM command
    port = 8000
    command = [
        "vllm",
        "serve",
        "--block-size",
        "16",
        "--port",
        str(port),
        "--kv-transfer-config",
        '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "kvbm.vllm_integration.connector"}',
        model,
        "--max-model-len",
        "8000",  # Required to fit on L4 GPU with smaller models
    ]

    # GPU blocks override
    if gpu_blocks is not None:
        command.extend(["--num-gpu-blocks-override", str(gpu_blocks)])

    # Set up environment
    env = os.environ.copy()
    env.update(
        {
            "RUST_BACKTRACE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "DYN_LOG": "debug",
            "DYN_KVBM_METRICS": "true",
            "DYN_KVBM_METRICS_PORT": "6880",
            # DynamoConnector connection settings
            "NATS_SERVER": "nats://localhost:4222",
            "ETCD_ENDPOINTS": "http://localhost:2379",
        }
    )

    # CPU cache blocks override via env
    if cpu_blocks is not None:
        env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_blocks)

    # Start server with ManagedProcess
    timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "600"))
    log_dir = f"{request.node.name}_vllm"

    with ManagedProcess(
        command=command,
        env=env,
        health_check_ports=[port, 6880],  # vLLM server + KVBM metrics
        timeout=timeout,
        display_output=True,
        terminate_existing=True,
        stragglers=["vllm"],
        straggler_commands=["vllm serve"],
        log_dir=log_dir,
    ) as proc:
        # Give KVBM connector extra time to fully initialize
        print("Waiting 5 seconds for KVBM connector to fully initialize...")
        time.sleep(5)

        # Create wrapper object for compatibility with existing test code
        class ServerWrapper:
            """Wrapper to maintain compatibility with LLMServerManager interface."""

            def __init__(self):
                self.base_url = f"http://localhost:{port}"
                self.server_type = server_type
                self.cpu_cache_blocks = cpu_blocks
                self.gpu_cache_blocks = gpu_blocks
                self.port = port
                self.proc = proc

        yield ServerWrapper()


class TestDeterminism:
    """Test class for determinism validation."""

    def _establish_baseline(self, tester, prompt: str, max_tokens: int) -> str:
        """Establish baseline response: warmup -> clear cache -> baseline."""
        print("\n" + "=" * 70)
        print("ESTABLISHING BASELINE (warmup -> clear cache -> baseline)")
        print("=" * 70)

        # Step 1: Warmup
        print("\nStep 1: Warmup request...")
        try:
            warmup_response = tester.make_request(
                prompt, max_tokens=max_tokens, temperature=0, seed=42
            )
            print(f"Warmup response: {warmup_response}")
        except Exception as e:
            pytest.fail(f"Warmup request failed: {e}")

        # Step 2: Clear cache
        print("\nStep 2: Clearing cache...")
        try:
            tester.reset_prefix_cache()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Warning: Cache reset failed: {e}")

        # Step 3: Baseline request
        print("\nStep 3: Baseline request (after cache clear)...")
        try:
            baseline_response = tester.make_request(
                prompt, max_tokens=max_tokens, temperature=0, seed=42
            )
            print(f"Baseline response: {baseline_response}")
            print("\n✓ Baseline established")
            print("=" * 70)
            return baseline_response
        except Exception as e:
            pytest.fail(f"Baseline request failed: {e}")

    def _start_benchmark(self, llm_server) -> tuple:
        """Start vllm bench in background.

        Returns:
            tuple: (process, file_handle, log_path)
        """
        import subprocess

        model = os.environ.get(
            "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        bench_cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--model",
            model,
            "--base-url",
            llm_server.base_url,
            "--dataset-name",
            "random",
            "--random-input-len",
            "4000",
            "--random-output-len",
            "180",
            "--max-concurrency",
            "7",
            "--num-prompts",
            "2000",
        ]

        print(f"\nStarting vllm bench: {' '.join(bench_cmd)}")
        bench_log = os.path.join(str(Path(".")), "vllm_bench_semantic.log")
        bench_file = open(bench_log, "w")
        bench_process = subprocess.Popen(
            bench_cmd,
            stdout=bench_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return bench_process, bench_file, bench_log

    def _wait_for_benchmark_activity(self, initial_offload: int) -> bool:
        """Wait for benchmark to start creating offload activity.

        Args:
            initial_offload: Initial offload block count to compare against

        Returns:
            bool: True if benchmark activity detected, False otherwise
        """
        print("\nWaiting for benchmark to start and create memory pressure...")
        max_wait = int(os.environ.get("KVBM_BENCH_STARTUP_WAIT", "120"))

        for wait_iteration in range(max_wait // 5):
            time.sleep(5)
            elapsed = (wait_iteration + 1) * 5

            try:
                current_metrics = fetch_kvbm_metrics()
                current_offload = current_metrics.get("kvbm_offload_blocks_d2h", 0)

                if current_offload > initial_offload:
                    offload_delta = current_offload - initial_offload
                    print(
                        f" Benchmark activity detected after {elapsed}s ({offload_delta} blocks offloaded)"
                    )
                    print("Waiting additional 10s for benchmark to fully ramp up...")
                    time.sleep(10)
                    return True
                else:
                    print(f" Waiting... ({elapsed}s elapsed, no offload activity yet)")
            except Exception as e:
                print(f"  Waiting... ({elapsed}s elapsed, metrics check failed: {e})")

        print(f" Warning: No benchmark activity detected after {max_wait}s")
        return False

    def _compare_with_baseline(
        self, response: str, baseline: str, min_similarity: float, request_num: int
    ) -> dict:
        """Compare response with baseline. Returns comparison result dict.

        Returns a dict with keys:
        - exact_match: bool - True if response exactly matches baseline
        - semantic_match: bool - True if semantically equivalent (includes exact matches)
        - similarity: float - Similarity score 0.0-1.0
        - reason: str - Explanation of the result
        - diverge_pos: int - Character position where divergence starts (if not matching)
        - approx_token: int - Approximate token position of divergence
        - context_before: str - Text context before divergence point
        - baseline_continues: str - How baseline continues after divergence
        - response_continues: str - How response continues after divergence
        - request_num: int - Request number
        - response: str - Full response text
        - baseline: str - Full baseline text
        """
        result = {
            "request_num": request_num,
            "exact_match": False,
            "semantic_match": False,
            "similarity": 0.0,
            "reason": "",
            "response": response,
            "baseline": baseline,
        }

        # Check for exact match
        if response == baseline:
            result["exact_match"] = True
            result["semantic_match"] = True
            result["similarity"] = 1.0
            result["reason"] = "Exact match"
            return result

        # Check semantic equivalence
        is_equivalent, similarity, reason = are_semantically_equivalent(
            baseline, response, min_similarity=min_similarity
        )
        result["similarity"] = similarity
        result["reason"] = reason

        if is_equivalent:
            result["semantic_match"] = True
        else:
            # Find divergence point for reporting
            diverge_pos = 0
            for j, (c1, c2) in enumerate(zip(baseline, response)):
                if c1 != c2:
                    diverge_pos = j
                    break
            else:
                diverge_pos = min(len(baseline), len(response))

            approx_token = diverge_pos // 4

            result["diverge_pos"] = diverge_pos
            result["approx_token"] = approx_token
            result["context_before"] = baseline[max(0, diverge_pos - 30) : diverge_pos]
            result["baseline_continues"] = baseline[diverge_pos : diverge_pos + 50]
            result["response_continues"] = response[diverge_pos : diverge_pos + 50]

        return result

    def _report_results(
        self,
        num_requests: int,
        exact_matches: int,
        semantic_matches: int,
        mismatches: list,
    ):
        """Print final test results."""
        print("\n" + "=" * 70)
        print("SEMANTIC DETERMINISM RESULTS")
        print("=" * 70)
        print(f"Total requests: {num_requests}")
        print(
            f"Exact matches: {exact_matches}/{num_requests} ({exact_matches/num_requests:.1%})"
        )
        print(
            f"Semantic matches: {semantic_matches}/{num_requests} ({semantic_matches/num_requests:.1%})"
        )
        print(
            f"Semantic divergence: {len(mismatches)}/{num_requests} ({len(mismatches)/num_requests:.1%})"
        )

        if mismatches:
            print(f"\n{'='*70}")
            print(f"NON-DETERMINISTIC RESPONSES ({len(mismatches)} total):")
            print(f"{'='*70}")
            for mismatch in mismatches:
                req_num = mismatch["request_num"]
                if "error" in mismatch:
                    print(f"\nRequest {req_num}: FAILED - {mismatch['error']}")
                else:
                    print(
                        f"\nRequest {req_num}: MISMATCH (similarity: {mismatch.get('similarity', 0):.1%})"
                    )
                    print(f"  Baseline: {mismatch.get('baseline', '')[:150]}...")
                    print(f"  Got:      {mismatch.get('response', '')[:150]}...")

            semantic_success_rate = (semantic_matches / num_requests) * 100
            min_success_rate = 80.0

            print(f"\n{'='*70}")
            print(f"SEMANTIC SUCCESS RATE: {semantic_success_rate:.1f}%")
            print(f"{'='*70}")
            print(f"Failed requests: {[m['request_num'] for m in mismatches]}")

            if semantic_success_rate < min_success_rate:
                pytest.fail(
                    f"Semantic determinism test failed!\n"
                    f"Semantic match rate: {semantic_success_rate:.1f}% (< {min_success_rate:.0f}%)\n"
                    f"This indicates significant non-determinism beyond FP precision effects"
                )
            else:
                print(
                    f"TEST PASSED - SEMANTICALLY DETERMINISTIC (>= {min_success_rate:.0f}%)"
                )
        else:
            print(f"\n{'='*70}")
            print("TEST PASSED - ALL RESPONSES SEMANTICALLY EQUIVALENT")
            print(f"{'='*70}")
            print(
                f"Exact matches: {exact_matches}/{num_requests} ({exact_matches/num_requests:.1%})"
            )

    def _show_final_kvbm_stats(self, initial_offload: int):
        """Display final KVBM metrics and compare with initial state.

        Args:
            initial_offload: Initial offload block count to compare against

        Raises:
            pytest.fail: If no offload activity was detected during the test
        """
        print(f"\n{'='*70}")
        print("FINAL KVBM STATS")
        print(f"{'='*70}")
        try:
            final_metrics = fetch_kvbm_metrics()
            final_offload = final_metrics.get("kvbm_offload_blocks_d2h", 0)
            final_onboard = final_metrics.get("kvbm_onboard_blocks_h2d", 0)

            offload_delta = final_offload - initial_offload
            print(f"Initial offload: {initial_offload} blocks")
            print(f"Final offload:   {final_offload} blocks")
            print(f"Total offloaded: {offload_delta} blocks")
            print(f"Total onboarded: {final_onboard} blocks")

            if offload_delta > 0:
                print(
                    f"\n KVBM offload activity detected: {offload_delta} blocks offloaded"
                )
            else:
                pytest.fail(
                    f"No offload activity detected during test.\n"
                    f"Initial offload: {initial_offload} blocks, Final offload: {final_offload} blocks.\n"
                    f"Test requires memory pressure to properly validate determinism under load."
                )

            if final_onboard > 0:
                print(
                    f" KVBM onboard activity detected: {final_onboard} blocks onboarded"
                )
            else:
                pytest.fail(
                    f"No onboard activity detected during test.\n"
                    f"Final onboard: {final_onboard} blocks.\n"
                    f"Test requires KV cache onboarding to properly validate determinism under load."
                )

        except Exception as e:
            print(f"Could not fetch final metrics: {e}")

    def base_test_spanish_prompt_determinism_under_load(
        self, tester, llm_server, runtime_services, spanish_prompt_path: Path
    ):
        """Base implementation: send Spanish prompt repeatedly while vllm bench runs.

        Tests determinism under high concurrency load. Reproduces bugs where responses
        can become corrupted or non-deterministic under memory pressure.

        Args:
            tester: DeterminismTester instance
            llm_server: LLM server manager
            runtime_services: Runtime services fixture
            spanish_prompt_path: Path to the Spanish prompt file
        """
        import subprocess

        print("\n" + "=" * 70)
        print("DETERMINISM TEST UNDER HIGH CONCURRENCY LOAD")
        print("=" * 70)

        # Load prompt
        prompt = load_prompt_from_file(spanish_prompt_path)
        if prompt is None:
            pytest.fail(f"Prompt not found at {spanish_prompt_path}")

        # Test parameters
        num_requests = int(os.environ.get("KVBM_NUM_ITERATIONS", "15"))
        delay_seconds = int(os.environ.get("KVBM_REQUEST_DELAY", "30"))
        max_tokens = int(os.environ.get("KVBM_MAX_TOKENS", "80"))
        min_similarity = float(os.environ.get("KVBM_MIN_SIMILARITY", "0.75"))

        print("\nTest configuration:")
        print(f"  Requests: {num_requests}")
        print(f"  Delay: {delay_seconds}s")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Min semantic similarity: {min_similarity:.0%}")

        # Establish baseline
        baseline_response = self._establish_baseline(tester, prompt, max_tokens)

        # Start benchmark
        bench_process, bench_file, bench_log = self._start_benchmark(llm_server)

        try:
            # Check initial metrics
            print("\nChecking initial KVBM metrics...")
            try:
                initial_metrics = fetch_kvbm_metrics()
                initial_offload = initial_metrics.get("kvbm_offload_blocks_d2h", 0)
                print(f"Initial offload: {initial_offload} blocks")
            except Exception as e:
                print(f"Could not fetch initial metrics: {e}")
                initial_offload = 0

            # Wait for benchmark activity
            benchmark_started = self._wait_for_benchmark_activity(initial_offload)
            if not benchmark_started:
                pytest.fail(
                    "Benchmark failed to start or create offload activity. "
                    "Test cannot proceed without memory pressure to properly test determinism under load."
                )

            print("Waiting additional 10s for benchmark to fully ramp up...")
            time.sleep(10)

            # Send requests and track results
            print(f"\n{'='*70}")
            print(f"SENDING {num_requests} REQUESTS (comparing against baseline)")
            print(f"{'='*70}")

            responses = []
            mismatches = []
            exact_matches = 0
            semantic_matches = 0

            for i in range(num_requests):
                print(f"\n--- Request {i+1}/{num_requests} ---")

                try:
                    response = tester.make_request(
                        prompt, max_tokens=max_tokens, temperature=0, seed=42
                    )
                    responses.append(response)
                    print(f"Response: {response}")

                    # Compare with baseline
                    comparison = self._compare_with_baseline(
                        response, baseline_response, min_similarity, i + 1
                    )

                    if comparison["exact_match"]:
                        print("✓ EXACT MATCH (100% deterministic)")
                        exact_matches += 1
                        semantic_matches += 1
                    elif comparison["semantic_match"]:
                        print(
                            f"✓ SEMANTICALLY EQUIVALENT ({comparison['similarity']:.1%} similar)"
                        )
                        print(f"  {comparison['reason']}")
                        semantic_matches += 1
                    else:
                        print(
                            f"✗ SEMANTIC DIVERGENCE ({comparison['similarity']:.1%} similar)"
                        )
                        print(f"  {comparison['reason']}")
                        print(
                            f"  Divergence at char {comparison['diverge_pos']} (~token {comparison['approx_token']})"
                        )
                        print(f"  Context before: ...{comparison['context_before']}")
                        print(
                            f"  Baseline continues: {comparison['baseline_continues']}..."
                        )
                        print(
                            f"  Response continues: {comparison['response_continues']}..."
                        )
                        mismatches.append(comparison)

                except Exception as e:
                    print(f"Request failed: {e}")
                    responses.append(None)
                    mismatches.append({"request_num": i + 1, "error": str(e)})

                # Wait before next request
                if i < num_requests - 1:
                    print(f"Waiting {delay_seconds}s...")
                    time.sleep(delay_seconds)

            # Report results
            self._report_results(
                num_requests, exact_matches, semantic_matches, mismatches
            )

            # Show final KVBM stats
            self._show_final_kvbm_stats(initial_offload)

        finally:
            print("\nStopping benchmark...")
            try:
                bench_process.terminate()
                bench_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                bench_process.kill()
                bench_process.wait()
            bench_file.close()
            print(f"Benchmark log: {bench_log}")

    def base_test_determinism_with_cache_reset(
        self, tester, llm_server, runtime_services, success_rate_threshold=1.0
    ):
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
        success_rate = (
            total_passed / (total_passed + total_failed)
            if total_passed + total_failed > 0
            else 0
        )
        print(f"Success rate: {success_rate:.1%}")
        print(
            "Test compared responses before cache reset (with warmup) vs after cache reset (no warmup)."
        )

        if total_passed + total_failed == 0:
            pytest.skip("No tests were completed - insufficient data")

        assert (
            success_rate >= success_rate_threshold
        ), f"Model is not deterministic across cache reset: {total_failed} comparisons failed, success rate {success_rate:.1%} lower than expected {success_rate_threshold*100}%"


# ============================================================================
# KVBM Test Helper Functions
# ============================================================================
# Note: KVBM fixtures are in conftest.py for automatic pytest discovery

KVBM_METRICS_PORT = 6880


def parse_kvbm_metrics(metrics_text: str) -> dict:
    """Parse KVBM metrics from Prometheus format.

    Args:
        metrics_text: Raw Prometheus metrics text

    Returns:
        Dictionary mapping metric names to integer values
    """
    metrics = {}
    for line in metrics_text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        for metric_name in [
            "kvbm_offload_blocks_d2h",
            "kvbm_onboard_blocks_h2d",
            "kvbm_offload_blocks_h2d",
            "kvbm_onboard_blocks_d2d",
            "kvbm_matched_tokens",
        ]:
            if line.startswith(metric_name + " "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    metrics[metric_name] = int(parts[1])
    return metrics


def fetch_kvbm_metrics(port: int = KVBM_METRICS_PORT, timeout: int = 10) -> dict:
    """Fetch and parse KVBM metrics from the metrics endpoint.

    Args:
        port: Metrics server port (default: 6880)
        timeout: Request timeout in seconds

    Returns:
        Dictionary of parsed metrics

    Raises:
        RuntimeError: If metrics endpoint is unreachable or returns error
    """
    response = requests.get(f"http://localhost:{port}/metrics", timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Metrics endpoint returned status {response.status_code}. "
            "Metrics server may not have started."
        )
    return parse_kvbm_metrics(response.text)


def assert_deterministic(
    response1: str,
    response2: str,
    test_name: str = "",
    label1: str = "Response 1",
    label2: str = "Response 2",
) -> None:
    """Verify two responses are identical (deterministic).

    Args:
        response1: First response text
        response2: Second response text
        test_name: Name of test for error messages
        label1: Label for first response in output
        label2: Label for second response in output

    Raises:
        pytest.fail: If responses differ
    """
    if response1 == response2:
        print(f" ✓ PASS: {test_name} responses are deterministic")
        print(f"    {label1}: {response1}")
        print(f"    {label2}: {response2}")
    else:
        print(f" ✗ FAIL: {test_name} responses differ")
        print(f"    {label1}: {response1}")
        print(f"    {label2}: {response2}")
        pytest.fail(
            f"{test_name}: Responses not deterministic\n"
            f"{label1}: {response1}\n"
            f"{label2}: {response2}"
        )

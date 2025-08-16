#!/usr/bin/env python3
"""
Simplified determinism test for language model API.
Tests if the model produces deterministic outputs with fixed seed and temperature=0.
Automatically starts and stops vLLM server for each test.
"""

import difflib
import logging
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests


class VLLMServerManager:
    """Manages vLLM server lifecycle."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        port: int = 8000,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        enforce_eager: bool = False,
    ):
        self.base_url = base_url
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks
        self.enforce_eager = enforce_eager

        # Setup logging directories
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)

        # Create unique log file names with timestamp and cache config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eager_str = "eager" if enforce_eager else "default"
        config_str = f"cpu{cpu_cache_blocks or 'default'}_gpu{gpu_cache_blocks or 'default'}_{eager_str}"
        self.server_log_file = (
            self.log_dir / f"vllm_server_{config_str}_{timestamp}.log"
        )
        self.server_stdout_file = None
        self.server_stderr_file = None

        # Build server command based on your configuration
        self.server_cmd = [
            "vllm",
            "serve",
            "--block-size",
            "16",
            "--port",
            str(port),
            "--kv-transfer-config",
            '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "dynamo.llm.vllm_integration.connector"}',
            "Qwen/Qwen3-0.6B",
        ]

        # Add enforce-eager flag if specified
        if enforce_eager:
            self.server_cmd.append("--enforce-eager")

        # Add GPU blocks override if specified
        if gpu_cache_blocks is not None:
            self.server_cmd.extend(["--num-gpu-blocks-override", str(gpu_cache_blocks)])

        # Build environment variables
        self.env = os.environ.copy()
        self.env.update(
            {
                "RUST_BACKTRACE": "1",
                "DYN_LOG": "debug,dynamo_llm::block_manager::layout=error",
                "VLLM_SERVER_DEV_MODE": "1",
            }
        )

        # Add CPU cache blocks override if specified
        if cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

    def start_server(self, timeout: int = 300) -> bool:
        """Start vLLM server and wait for it to be ready."""
        if self.is_server_running():
            print("Server already running, stopping it first...")
            self.stop_server()
            time.sleep(2)

        print(f"Starting vLLM server with command: {' '.join(self.server_cmd)}")
        print(
            f"Environment: CPU blocks={self.cpu_cache_blocks}, GPU blocks={self.gpu_cache_blocks}, Eager={self.enforce_eager}"
        )
        print(f"Server logs will be saved to: {self.server_log_file}")

        # Open log files for server output
        self.server_stdout_file = open(
            self.server_log_file.with_suffix(".stdout.log"), "w"
        )
        self.server_stderr_file = open(
            self.server_log_file.with_suffix(".stderr.log"), "w"
        )

        # Write server startup info to log
        self.server_stdout_file.write(
            f"=== vLLM Server Started at {datetime.now()} ===\n"
        )
        self.server_stdout_file.write(f"Command: {' '.join(self.server_cmd)}\n")
        self.server_stdout_file.write(
            f"CPU blocks: {self.cpu_cache_blocks}, GPU blocks: {self.gpu_cache_blocks}, Eager: {self.enforce_eager}\n"
        )
        self.server_stdout_file.write("=" * 60 + "\n\n")
        self.server_stdout_file.flush()

        # Start server process with custom environment and logging
        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=self.server_stdout_file,
            stderr=self.server_stderr_file,
            env=self.env,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                print(f"vLLM server is ready at {self.base_url}")
                return True

            # Check if process died
            if self.process.poll() is not None:
                print(f"Server process died. Check logs at {self.server_log_file}")
                self._close_log_files()
                return False

            print("Waiting for server to start...")
            time.sleep(5)

        print(f"Server failed to start within {timeout} seconds")
        self.stop_server()
        return False

    def stop_server(self):
        """Stop vLLM server."""
        if self.process:
            print("Stopping vLLM server...")
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Server didn't stop gracefully, forcing shutdown...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()

                print("vLLM server stopped")
            except (ProcessLookupError, OSError):
                print("Server process already stopped")
            finally:
                self.process = None

        # Close log files
        self._close_log_files()

    def _close_log_files(self):
        """Close server log files."""
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
        """Check if vLLM server is running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class SimpleDeterminismTester:
    """Simplified determinism tester."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

        # Setup test logging
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_log_file = self.log_dir / f"test_execution_{timestamp}.log"

        # Setup logger
        self.logger = logging.getLogger("DeterminismTester")
        self.logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler for detailed logs
        file_handler = logging.FileHandler(self.test_log_file)
        file_handler.setLevel(logging.DEBUG)  # Keep detailed logs in file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for important messages only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.WARNING
        )  # Only show warnings and errors on console
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(
            f"Test logging initialized. Logs will be saved to: {self.test_log_file}"
        )

        # Test prompts of varying lengths and complexity
        self.test_prompts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog. This is a standard pangram that contains all letters of the alphabet.",
            "Find light in the beautiful sea, I choose to be happy, You and I, you and I, we are like a beautiful melody that never ends, dancing through the night with stars as our companions, whispering secrets to the wind as we journey through life together, hand in hand, heart to heart, forever and always.",
            "The advancement of technology has fundamentally transformed the way we live, work, and communicate in the modern world. From the invention of the printing press to the development of the internet, each technological breakthrough has opened new possibilities and created unprecedented opportunities for human progress. Today, artificial intelligence and machine learning are reshaping industries, healthcare, education, and countless other fields, promising to solve complex problems and improve the quality of life for people around the globe.",
            "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
            "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons, each connected to thousands of others through intricate networks of synapses. This biological supercomputer processes information at speeds that would make even the most advanced artificial intelligence systems seem primitive by comparison. Every thought, memory, emotion, and decision we make is the result of electrical and chemical signals traveling through this vast neural network. The brain's ability to learn, adapt, and create is unmatched by any machine we have ever built. It can recognize patterns in milliseconds, solve complex problems through intuition, and generate creative ideas that have never existed before. Yet despite our incredible advances in neuroscience, we still understand only a fraction of how this remarkable organ truly works. The mysteries of consciousness, memory formation, and the nature of human intelligence continue to challenge the brightest minds in science and philosophy.",
            "The cat sat on the mat while the dog slept peacefully in the corner, creating a perfect picture of domestic tranquility that warmed the heart of anyone who witnessed this simple moment of harmony between two natural enemies turned friends.",
            "Mathematics is the language of the universe, and numbers are its alphabet. Through the elegant dance of equations and the symphony of algorithms, we unlock the secrets of nature's most profound mysteries. From the simple beauty of prime numbers to the complex elegance of calculus, mathematics provides us with the tools to understand everything from the smallest subatomic particles to the vast expanse of galaxies stretching across the cosmic void.",
            "A journey of a thousand miles begins with a single step, as the ancient Chinese proverb wisely reminds us. This timeless wisdom speaks to the fundamental truth that every great achievement, every monumental discovery, and every life-changing transformation starts with that crucial moment of decision - the moment when we choose to take action instead of remaining in the comfort of inaction. Whether it's learning a new skill, starting a business, writing a novel, or embarking on a spiritual quest, the path to success is paved with countless small steps, each one building upon the last, until we find ourselves transformed by the journey itself.",
            "Technology evolves rapidly, but human nature remains constant through the ages. Despite the incredible advances in artificial intelligence, virtual reality, and biotechnology, the fundamental desires, fears, and aspirations that drive human behavior have remained remarkably consistent throughout history. We still seek connection, meaning, and purpose in our lives. We still fear the unknown and crave security. We still dream of a better future and work to create it for ourselves and our loved ones. This paradox - the ever-changing nature of our tools and the unchanging nature of our hearts - is perhaps the most fascinating aspect of the human condition, reminding us that while we may build increasingly sophisticated machines, we remain fundamentally human in our core essence.",
        ]

    def make_request(self, content: str) -> str:
        """Make API request and return completion text."""
        payload = {
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_completion_tokens": 24,
            "temperature": 0,
            "top_p": 0.0001,
            "seed": 42,
        }

        # Only log payload to file, not console
        self.logger.debug(f"Making request with payload: {payload}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            result = data["choices"][0]["message"]["content"]

            return result

        except Exception as e:
            self.logger.error(
                f"Request failed for content '{content[:50]}...': {str(e)}"
            )
            raise

    def reset_cache(self):
        """Reset the prefix cache."""
        self.logger.info("Resetting prefix cache")
        try:
            response = requests.post(f"{self.base_url}/reset_prefix_cache", timeout=10)
            response.raise_for_status()
            self.logger.info("Cache reset successful")
        except Exception as e:
            self.logger.error(f"Cache reset failed: {str(e)}")
            raise

    def _calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate three core text similarity metrics between two strings."""
        if not text1 or not text2:
            return {
                "sequence_similarity": 0.0,
                "word_similarity": 0.0,
                "character_similarity": 0.0,
                "composite_similarity": 0.0,
            }

        # Sequence similarity using difflib
        sequence_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

        # Word-level similarity
        words1 = text1.split()
        words2 = text2.split()
        word_similarity = difflib.SequenceMatcher(None, words1, words2).ratio()

        # Character-level similarity (ignoring whitespace)
        chars1 = re.sub(r"\s+", "", text1.lower())
        chars2 = re.sub(r"\s+", "", text2.lower())
        char_similarity = difflib.SequenceMatcher(None, chars1, chars2).ratio()

        # Composite similarity: weighted average of the three metrics
        # Sequence similarity gets highest weight as it's most comprehensive
        # Word similarity gets medium weight as it captures semantic structure
        # Character similarity gets lowest weight as it's most sensitive to minor changes
        composite_similarity = (
            0.5 * sequence_similarity
            + 0.3 * word_similarity  # 50% weight
            + 0.2 * char_similarity  # 30% weight  # 20% weight
        )

        return {
            "sequence_similarity": round(sequence_similarity, 4),
            "word_similarity": round(word_similarity, 4),
            "character_similarity": round(char_similarity, 4),
            "composite_similarity": round(composite_similarity, 4),
        }

    def _calculate_semantic_severity(
        self, text1: str, text2: str, analysis: Dict
    ) -> Dict[str, any]:
        """Calculate semantic severity based on the three core similarity metrics."""
        # Extract the different parts
        divergence_point = analysis["divergence_point"]

        # Get the divergent parts
        divergent1 = text1[divergence_point:].strip()
        divergent2 = text2[divergence_point:].strip()

        # Start with base severity
        severity_score = 0.0
        severity_reasons = []

        # Get similarity metrics
        metrics = analysis["similarity_metrics"]
        composite_sim = metrics["composite_similarity"]
        sequence_sim = metrics["sequence_similarity"]
        word_sim = metrics["word_similarity"]
        char_sim = metrics["character_similarity"]

        # 1. Primary severity scoring based on composite similarity (adjusted thresholds)
        if composite_sim > 0.95:
            severity_score += 0.2  # Very minor difference
            severity_reasons.append("Very high composite similarity")
        elif composite_sim > 0.90:
            severity_score += 0.5  # Minor difference
            severity_reasons.append("High composite similarity")
        elif composite_sim > 0.85:
            severity_score += 1.0  # Moderate difference
            severity_reasons.append("Good composite similarity")
        elif composite_sim > 0.75:
            severity_score += 1.5  # Noticeable difference
            severity_reasons.append("Moderate composite similarity")
        elif composite_sim > 0.65:
            severity_score += 2.5  # Significant difference
            severity_reasons.append("Low composite similarity")
        elif composite_sim > 0.50:
            severity_score += 3.5  # Major difference
            severity_reasons.append("Very low composite similarity")
        else:
            severity_score += 4.5  # Severe difference
            severity_reasons.append("Extremely low composite similarity")

        # 2. Additional penalty for individual metric failures (reduced penalties)
        low_metrics = []
        if sequence_sim < 0.6:  # More lenient threshold
            low_metrics.append("sequence")
        if word_sim < 0.5:  # More lenient threshold
            low_metrics.append("word")
        if char_sim < 0.7:  # More lenient threshold
            low_metrics.append("character")

        if low_metrics:
            severity_score += 0.3 * len(low_metrics)  # Reduced penalty
            severity_reasons.append(f"Low {', '.join(low_metrics)} similarity")

        # 3. Length-based adjustments (reduced impact)
        total_length = max(len(text1), len(text2))
        divergent_length1 = len(divergent1)
        divergent_length2 = len(divergent2)
        max_divergent_length = max(divergent_length1, divergent_length2)

        divergent_ratio = max_divergent_length / total_length if total_length > 0 else 0

        # Only penalize if most of the text is truly different
        if divergent_ratio > 0.8:  # Raised threshold
            severity_score += 0.8  # Reduced penalty
            severity_reasons.append("Large portion differs")
        elif divergent_ratio > 0.6:  # Raised threshold
            severity_score += 0.4  # Reduced penalty
            severity_reasons.append("Moderate portion differs")

        # 4. Early divergence penalty (reduced impact)
        if total_length > 0:
            early_divergence_factor = 1 - (divergence_point / total_length)
            if divergence_point < 5:  # Very early (stricter threshold)
                severity_score += 0.8  # Reduced penalty
                severity_reasons.append("Very early divergence")
            elif divergence_point < 15:  # Early (stricter threshold)
                severity_score += 0.3  # Reduced penalty
                severity_reasons.append("Early divergence")
        else:
            early_divergence_factor = 0

        # 5. Critical semantic content analysis (more nuanced)
        words1 = set(divergent1.lower().split())
        words2 = set(divergent2.lower().split())

        # Look for truly critical differences with more context
        critical_shifts = {
            "subject_change": (
                ["i", "me", "my"],
                ["you", "your", "they", "them", "user"],
            ),
            "action_change": (
                ["want", "need", "should"],
                ["wants", "needs", "requesting"],
            ),
            "negation_change": (
                ["not", "dont", "don't", "never", "cannot", "cant"],
                ["can", "will", "should", "yes"],
            ),
        }

        critical_shift_penalty = 0
        for shift_type, (words_group1, words_group2) in critical_shifts.items():
            has_group1_text1 = any(word in words1 for word in words_group1)
            has_group2_text1 = any(word in words1 for word in words_group2)
            has_group1_text2 = any(word in words2 for word in words_group1)
            has_group2_text2 = any(word in words2 for word in words_group2)

            # Only flag if there's a clear shift from one group to another
            if (has_group1_text1 and has_group2_text2) or (
                has_group2_text1 and has_group1_text2
            ):
                # But check if this is actually meaningful in context
                # For subject_change, check if it's just pronoun variation vs actual perspective shift
                if shift_type == "subject_change":
                    # If both responses are talking about the same thing but with different pronouns,
                    # this might be less severe than a true perspective change
                    divergent_lower1 = divergent1.lower()
                    divergent_lower2 = divergent2.lower()

                    # Check for similar sentence structure (indicates minor pronoun difference)
                    if (
                        ("find" in divergent_lower1 and "find" in divergent_lower2)
                        or ("user" in divergent_lower1 and "user" in divergent_lower2)
                        or (
                            len(
                                set(divergent_lower1.split())
                                & set(divergent_lower2.split())
                            )
                            > 3
                        )
                    ):
                        critical_shift_penalty += (
                            0.5  # Reduced penalty for contextual pronoun changes
                        )
                        severity_reasons.append(f"Minor {shift_type}")
                    else:
                        critical_shift_penalty += (
                            1.2  # Full penalty for true perspective changes
                        )
                        severity_reasons.append(f"Critical {shift_type}")
                else:
                    critical_shift_penalty += (
                        1.0  # Normal penalty for other critical changes
                    )
                    severity_reasons.append(f"Critical {shift_type}")

        severity_score += critical_shift_penalty

        # 6. Check for minor word substitutions (should significantly reduce severity)
        common_words = words1.intersection(words2)
        total_unique_words = len(words1.union(words2))
        word_overlap_ratio = (
            len(common_words) / total_unique_words if total_unique_words > 0 else 0
        )

        # More aggressive reduction for high word overlap
        if word_overlap_ratio > 0.85:
            severity_score *= 0.5  # Reduce by 50% for very high overlap
            severity_reasons.append("Very high word overlap")
        elif word_overlap_ratio > 0.75:
            severity_score *= 0.65  # Reduce by 35% for high overlap
            severity_reasons.append("High word overlap")
        elif word_overlap_ratio > 0.6:
            severity_score *= 0.8  # Reduce by 20% for moderate overlap
            severity_reasons.append("Moderate word overlap")

        # 7. Check for simple rephrasing patterns
        # If the core meaning words are preserved, reduce severity
        important_content_words1 = {
            word
            for word in words1
            if len(word) > 3 and word not in {"that", "they", "them", "this", "with"}
        }
        important_content_words2 = {
            word
            for word in words2
            if len(word) > 3 and word not in {"that", "they", "them", "this", "with"}
        }

        content_overlap = len(important_content_words1 & important_content_words2)
        total_content_words = len(important_content_words1 | important_content_words2)
        content_overlap_ratio = (
            content_overlap / total_content_words if total_content_words > 0 else 0
        )

        if content_overlap_ratio > 0.8:
            severity_score *= 0.7  # Reduce severity for preserved content words
            severity_reasons.append("Core content preserved")

        # Normalize severity score (0-10 scale)
        severity_score = min(severity_score, 10.0)

        # Determine severity level with adjusted thresholds
        if severity_score >= 4.0:
            severity_level = "SEVERE"
        elif severity_score >= 2.5:
            severity_level = "MODERATE"
        elif severity_score >= 1.0:
            severity_level = "MINOR"
        else:
            severity_level = "TRIVIAL"

        return {
            "severity_score": round(severity_score, 2),
            "severity_level": severity_level,
            "severity_reasons": severity_reasons,
            "divergent_ratio": round(divergent_ratio, 3),
            "early_divergence_factor": round(early_divergence_factor, 3),
            "divergent_text1": divergent1,
            "divergent_text2": divergent2,
            "semantic_analysis": {
                "different_words_count": len(words1.symmetric_difference(words2)),
                "word_overlap_ratio": round(word_overlap_ratio, 3),
                "content_overlap_ratio": round(content_overlap_ratio, 3),
                "composite_similarity_used": True,
                "critical_shifts": len(
                    [r for r in severity_reasons if "Critical" in r]
                ),
            },
        }

    def _analyze_response_differences(self, text1: str, text2: str) -> Dict[str, any]:
        """Analyze the differences between two responses in detail."""
        # Get similarity metrics
        similarity = self._calculate_text_similarity(text1, text2)

        # Get detailed diff
        diff_lines = list(
            difflib.unified_diff(
                text1.splitlines(keepends=True),
                text2.splitlines(keepends=True),
                fromfile="before_reset",
                tofile="after_reset",
                lineterm="",
            )
        )

        # Count different types of changes
        additions = sum(
            1
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1
            for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        )

        # Find common prefix and suffix
        common_prefix = os.path.commonprefix([text1, text2])
        common_suffix = os.path.commonprefix([text1[::-1], text2[::-1]])[::-1]

        analysis = {
            "similarity_metrics": similarity,
            "diff_lines": diff_lines,
            "additions": additions,
            "deletions": deletions,
            "common_prefix": common_prefix,
            "common_suffix": common_suffix,
            "common_prefix_length": len(common_prefix),
            "common_suffix_length": len(common_suffix),
            "divergence_point": len(common_prefix),
        }

        # Add semantic severity analysis
        severity = self._calculate_semantic_severity(text1, text2, analysis)
        analysis["semantic_severity"] = severity

        return analysis

    def test_determinism_with_cache_reset(self, prompts: List[str]) -> bool:
        """Test determinism by comparing responses before and after cache reset."""
        self.logger.info(f"Starting determinism test with {len(prompts)} prompts")

        print("Phase 1: Getting responses BEFORE cache reset")
        print("-" * 50)
        self.logger.info("Phase 1: Getting responses BEFORE cache reset")

        # Store responses from first phase
        phase1_responses = {}

        for i, prompt in enumerate(prompts):
            print(f"Testing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            self.logger.info(f"Testing prompt {i+1}/{len(prompts)}: {prompt}")
            response = self.make_request(prompt)
            phase1_responses[prompt] = response
            print(f"  Response: {response}")

        print(f"\nPhase 1 Complete: Collected {len(phase1_responses)} responses")

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        self.reset_cache()

        # Phase 2: Test same prompts after cache reset
        print("\nPhase 2: Getting responses AFTER cache reset")
        print("-" * 50)
        self.logger.info("Phase 2: Getting responses AFTER cache reset")

        phase2_responses = {}
        inconsistent_prompts = []
        similarity_analyses = (
            {}
        )  # Store detailed similarity analysis for failed prompts

        for i, prompt in enumerate(prompts):
            print(f"Testing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            response = self.make_request(prompt)
            phase2_responses[prompt] = response
            print(f"  Response: {response}")

            # Check consistency across cache reset
            phase1_response = phase1_responses[prompt]

            if phase1_response != response:
                print("    INCONSISTENT across cache reset")
                print(f"    Before reset: {phase1_response}")
                print(f"    After reset:  {response}")

                # Perform detailed similarity analysis
                analysis = self._analyze_response_differences(phase1_response, response)
                similarity_analyses[prompt] = analysis

                # Show similarity metrics
                metrics = analysis["similarity_metrics"]
                severity = analysis["semantic_severity"]  # Add this line back
                similarity_display = (
                    f"Seq={metrics['sequence_similarity']:.3f}, "
                    f"Word={metrics['word_similarity']:.3f}, "
                    f"Char={metrics['character_similarity']:.3f}, "
                    f"Comp={metrics['composite_similarity']:.3f}"
                )

                # if metrics['vector_similarity'] > 0:
                #     similarity_display += f", Vector={metrics['vector_similarity']:.3f}"

                print(f"    Similarity: {similarity_display}")
                print(
                    f"    Semantic Severity: {severity['severity_level']} (Score: {severity['severity_score']}/10)"
                )
                print(f"    Divergence at position: {analysis['divergence_point']}")

                inconsistent_prompts.append(prompt)
                self.logger.warning(
                    f"INCONSISTENT for prompt '{prompt[:50]}...': "
                    f"Composite={metrics['composite_similarity']:.3f}, "
                    f"Severity={severity['severity_level']}"
                )

        # Final results

        print(f"\nPhase 2 Complete: Collected {len(phase2_responses)} responses")

        # Enhanced inconsistent prompts summary with similarity analysis
        if inconsistent_prompts:
            print("\n" + "=" * 80)
            print(
                f"INCONSISTENT PROMPTS WITH SIMILARITY ANALYSIS ({len(inconsistent_prompts)}):"
            )
            print("=" * 80)

            # Sort by semantic severity first, then by similarity
            sorted_prompts = sorted(
                inconsistent_prompts,
                key=lambda p: (
                    -similarity_analyses[p]["semantic_severity"][
                        "severity_score"
                    ],  # Most severe first (negative for desc)
                    -similarity_analyses[p]["similarity_metrics"][
                        "sequence_similarity"
                    ],  # Then most similar
                ),
            )

            for i, prompt in enumerate(sorted_prompts):
                analysis = similarity_analyses[prompt]
                metrics = analysis["similarity_metrics"]
                severity = analysis["semantic_severity"]

                print(f"\n{i+1}. Prompt: {prompt[:60]}...")
                print(
                    f"   SEMANTIC SEVERITY: {severity['severity_level']} (Score: {severity['severity_score']}/10)"
                )
                if severity["severity_reasons"]:
                    print(
                        f"   Severity Reasons: {', '.join(severity['severity_reasons'])}"
                    )

                print("   Similarity Metrics:")
                print(
                    f"     • Sequence Similarity: {metrics['sequence_similarity']:.4f}"
                )
                print(f"     • Word Similarity: {metrics['word_similarity']:.4f}")
                print(
                    f"     • Character Similarity: {metrics['character_similarity']:.4f}"
                )
                print(
                    f"     • Composite Similarity: {metrics['composite_similarity']:.4f}"
                )
                print("   Difference Analysis:")
                print(
                    f"     • Common prefix length: {analysis['common_prefix_length']} chars"
                )
                print(
                    f"     • Divergence point: position {analysis['divergence_point']}"
                )
                print(
                    f"     • Divergent content ratio: {severity['divergent_ratio']:.3f}"
                )
                print(
                    f"     • Early divergence factor: {severity['early_divergence_factor']:.3f}"
                )
                print(
                    f"     • Additions: {analysis['additions']}, Deletions: {analysis['deletions']}"
                )

                # Show semantic analysis details
                semantic = severity["semantic_analysis"]
                print("   Semantic Analysis:")
                if "different_words_count" in semantic.keys():
                    print(
                        f"     • Different words: {semantic['different_words_count']}"
                    )
                if "critical_word_changes" in semantic.keys():
                    print(
                        f"     • Critical word changes: {semantic['critical_word_changes']}"
                    )
                if "interpretation_changes" in semantic.keys():
                    print(
                        f"     • Interpretation changes: {semantic['interpretation_changes']}"
                    )

                # Show the actual differences
                print("   Responses:")
                print(f"     Before: {phase1_responses[prompt]}")
                print(f"     After:  {phase2_responses[prompt]}")

                # Show divergent parts specifically
                print("   Divergent Parts:")
                print(f"     Before: '{severity['divergent_text1']}'")
                print(f"     After:  '{severity['divergent_text2']}'")

                # Show common prefix if it exists
                if analysis["common_prefix"]:
                    print(
                        f"   Common prefix: '{analysis['common_prefix'][:100]}{'...' if len(analysis['common_prefix']) > 100 else ''}'"
                    )

            # Enhanced summary statistics
            print("\n" + "-" * 80)
            print("ENHANCED SIMILARITY & SEVERITY STATISTICS:")
            print("-" * 80)

            sequence_similarities = [
                similarity_analyses[p]["similarity_metrics"]["sequence_similarity"]
                for p in inconsistent_prompts
            ]
            word_similarities = [
                similarity_analyses[p]["similarity_metrics"]["word_similarity"]
                for p in inconsistent_prompts
            ]
            char_similarities = [
                similarity_analyses[p]["similarity_metrics"]["character_similarity"]
                for p in inconsistent_prompts
            ]
            composite_similarities = [
                similarity_analyses[p]["similarity_metrics"]["composite_similarity"]
                for p in inconsistent_prompts
            ]
            severity_scores = [
                similarity_analyses[p]["semantic_severity"]["severity_score"]
                for p in inconsistent_prompts
            ]

            print("Similarity Metrics Summary:")
            print(
                f"  Sequence Similarity - Min: {min(sequence_similarities):.4f}, "
                f"Max: {max(sequence_similarities):.4f}, "
                f"Avg: {sum(sequence_similarities)/len(sequence_similarities):.4f}"
            )
            print(
                f"  Word Similarity - Min: {min(word_similarities):.4f}, "
                f"Max: {max(word_similarities):.4f}, "
                f"Avg: {sum(word_similarities)/len(word_similarities):.4f}"
            )
            print(
                f"  Character Similarity - Min: {min(char_similarities):.4f}, "
                f"Max: {max(char_similarities):.4f}, "
                f"Avg: {sum(char_similarities)/len(char_similarities):.4f}"
            )
            print(
                f"  Composite Similarity - Min: {min(composite_similarities):.4f}, "
                f"Max: {max(composite_similarities):.4f}, "
                f"Avg: {sum(composite_similarities)/len(composite_similarities):.4f}"
            )
            print(
                f"Semantic Severity - Min: {min(severity_scores):.2f}, "
                f"Max: {max(severity_scores):.2f}, "
                f"Avg: {sum(severity_scores)/len(severity_scores):.2f}"
            )

            # Categorize failures by composite similarity level (our new primary metric)
            high_similarity = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["similarity_metrics"]["composite_similarity"]
                > 0.8
            ]
            medium_similarity = [
                p
                for p in inconsistent_prompts
                if 0.5
                < similarity_analyses[p]["similarity_metrics"]["composite_similarity"]
                <= 0.8
            ]
            low_similarity = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["similarity_metrics"]["composite_similarity"]
                <= 0.5
            ]

            # Categorize failures by semantic severity
            severe_failures = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["semantic_severity"]["severity_level"]
                == "SEVERE"
            ]
            moderate_failures = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["semantic_severity"]["severity_level"]
                == "MODERATE"
            ]
            minor_failures = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["semantic_severity"]["severity_level"]
                == "MINOR"
            ]
            trivial_failures = [
                p
                for p in inconsistent_prompts
                if similarity_analyses[p]["semantic_severity"]["severity_level"]
                == "TRIVIAL"
            ]

            print("\nComposite Similarity Categories:")
            print(f"  • High similarity (>80%): {len(high_similarity)} prompts")
            print(f"  • Medium similarity (50-80%): {len(medium_similarity)} prompts")
            print(f"  • Low similarity (<50%): {len(low_similarity)} prompts")

            print("\nSemantic Severity Categories:")
            print(f"  SEVERE failures: {len(severe_failures)} prompts")
            print(f"  MODERATE failures: {len(moderate_failures)} prompts")
            print(f"  MINOR failures: {len(minor_failures)} prompts")
            print(f"  TRIVIAL failures: {len(trivial_failures)} prompts")

            print("=" * 80)

        success = len(severe_failures) + len(moderate_failures) == 0
        print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")

        return success


@pytest.fixture(scope="function")
def vllm_server(request):
    """Start and stop vLLM server for each test with parameterized cache blocks."""
    # Setup pytest logging
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    # Get cache block parameters from test parametrization
    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    enforce_eager = getattr(request, "param", {}).get("enforce_eager", False)

    logger.info(
        f"Setting up vLLM server with CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
    )

    server_manager = VLLMServerManager(
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        enforce_eager=enforce_eager,
    )

    # Start server
    if not server_manager.start_server():
        logger.error(
            f"Failed to start vLLM server with CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
        )
        pytest.fail(
            f"Failed to start vLLM server with CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
        )

    logger.info("vLLM server started successfully")
    yield server_manager

    # Cleanup: stop server
    logger.info("Stopping vLLM server")
    server_manager.stop_server()
    logger.info("vLLM server stopped")


@pytest.fixture(scope="function")
def tester(vllm_server):
    """Create determinism tester with running server."""
    return SimpleDeterminismTester()


class TestDeterminism:
    """Determinism test cases."""

    @pytest.mark.parametrize(
        "vllm_server",
        [
            {"cpu_blocks": 10000, "enforce_eager": True},
            {"cpu_blocks": 10000, "enforce_eager": False},
        ],
        indirect=True,
    )
    def test_determinism_with_cache_reset(self, tester, vllm_server):
        """Test determinism before and after cache reset with same requests across different cache configurations."""
        cpu_blocks = vllm_server.cpu_cache_blocks
        gpu_blocks = vllm_server.gpu_cache_blocks
        enforce_eager = vllm_server.enforce_eager
        print(
            f"\nTesting with CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
        )

        # Log test start
        tester.logger.info("=== Starting test_determinism_with_cache_reset ===")
        tester.logger.info(
            f"Server configuration: CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
        )
        tester.logger.info(f"Server logs: {vllm_server.server_log_file}")

        try:
            result = tester.test_determinism_with_cache_reset(tester.test_prompts)
            tester.logger.info("=== Test completed successfully ===")
            assert (
                result
            ), f"Model responses are not deterministic with CPU blocks={cpu_blocks}, GPU blocks={gpu_blocks}, Eager={enforce_eager}"
        except Exception as e:
            tester.logger.error(f"=== Test failed with exception: {str(e)} ===")
            raise


def print_log_summary():
    """Print a summary of where logs are stored."""
    log_dir = Path("test_logs")
    if log_dir.exists():
        print("\n" + "=" * 60)
        print("LOG FILES SUMMARY")
        print("=" * 60)

        server_logs = list(log_dir.glob("vllm_server_*.log"))
        test_logs = list(log_dir.glob("test_execution_*.log"))

        print(f"Log directory: {log_dir.absolute()}")
        print(f"Server logs ({len(server_logs)} files):")
        for log_file in sorted(server_logs):
            print(f"  - {log_file.name}")

        print(f"Test execution logs ({len(test_logs)} files):")
        for log_file in sorted(test_logs):
            print(f"  - {log_file.name}")
        print("=" * 60)


if __name__ == "__main__":
    try:
        print("Starting determinism test...")
        print("=" * 60)

        # Create a simple test configuration - you can modify these values
        test_config = {
            "cpu_blocks": 10000,
            "gpu_blocks": None,
            "enforce_eager": True,  # Set to True to enable --enforce-eager flag
        }

        print(
            f"Test configuration: CPU blocks={test_config['cpu_blocks']}, GPU blocks={test_config['gpu_blocks']}, Eager={test_config['enforce_eager']}"
        )

        # Create server manager
        server_manager = VLLMServerManager(
            cpu_cache_blocks=test_config["cpu_blocks"],
            gpu_cache_blocks=test_config["gpu_blocks"],
            enforce_eager=test_config["enforce_eager"],
        )

        # Start server
        print("Starting vLLM server...")
        if not server_manager.start_server():
            print("ERROR: Failed to start vLLM server")
            exit(1)

        try:
            # Create tester
            tester = SimpleDeterminismTester()

            print(f"Server logs: {server_manager.server_log_file}")
            print(f"Test logs: {tester.test_log_file}")

            # Run the test
            result = tester.test_determinism_with_cache_reset(tester.test_prompts)

            if result:
                print("\n" + "=" * 60)
                print("TEST RESULT: PASS - All determinism tests passed!")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("TEST RESULT: FAIL - Some determinism tests failed!")
                print("=" * 60)
                exit(1)

        finally:
            # Always stop the server
            print("\nStopping vLLM server...")
            server_manager.stop_server()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if "server_manager" in locals():
            server_manager.stop_server()
        exit(1)
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        if "server_manager" in locals():
            server_manager.stop_server()
        exit(1)
    finally:
        print_log_summary()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: Stateful vs Stateless Responses API

Compares the performance characteristics of stateful (using `previous_response_id`)
vs stateless (full conversation history) approaches when using the OpenAI Responses API.

Key metrics:
- Request payload size
- Latency (total and time-to-first-token for streaming)
- Tokens sent per request

Usage:
    python benchmark_stateful_responses.py --base-url http://localhost:9000/v1 --model gpt-4
    python benchmark_stateful_responses.py --base-url http://localhost:9000/v1 --dry-run
    python benchmark_stateful_responses.py --base-url http://localhost:9000/v1 --turns 25

Author: Benchmark Script for Dynamo Responses API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: uv pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOKENS = 100

# Simulated conversation turns for benchmarking
CONVERSATION_PROMPTS = [
    "Hello! Can you help me understand machine learning?",
    "What's the difference between supervised and unsupervised learning?",
    "Can you give me an example of supervised learning?",
    "How does a neural network work?",
    "What are the main components of a neural network?",
    "Explain backpropagation in simple terms.",
    "What is gradient descent?",
    "What's the difference between batch and stochastic gradient descent?",
    "How do I prevent overfitting in my models?",
    "What is regularization?",
    "Explain L1 vs L2 regularization.",
    "What is dropout and how does it help?",
    "What are convolutional neural networks good for?",
    "How do CNNs process images?",
    "What is a pooling layer?",
    "What are recurrent neural networks?",
    "Explain LSTMs and why they're useful.",
    "What is the attention mechanism?",
    "How do transformers work?",
    "What makes GPT models different from earlier architectures?",
    "What is fine-tuning and when should I use it?",
    "How do I evaluate my model's performance?",
    "What metrics should I use for classification?",
    "Explain precision vs recall.",
    "What is the F1 score?",
]

# Simulated assistant responses (used in dry-run mode)
SIMULATED_RESPONSES = [
    "Of course! Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without predefined categories.",
    "A classic example is email spam classification, where the model learns from emails labeled as 'spam' or 'not spam' to categorize new emails.",
    "A neural network is composed of layers of interconnected nodes (neurons) that process information using weighted connections and activation functions.",
    "The main components are: input layer (receives data), hidden layers (process information), output layer (produces results), weights, biases, and activation functions.",
    "Backpropagation calculates how much each weight contributed to the error, then adjusts weights backward from output to input to minimize the error.",
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction that reduces the loss function most rapidly.",
    "Batch uses all data per update (stable but slow), while stochastic uses one sample (fast but noisy). Mini-batch is a practical middle ground.",
    "Use techniques like: cross-validation, regularization, dropout, early stopping, data augmentation, and keeping your model appropriately sized.",
    "Regularization adds a penalty term to the loss function to discourage overly complex models, helping prevent overfitting to training data.",
    "L1 (Lasso) promotes sparsity by pushing weights to zero, useful for feature selection. L2 (Ridge) penalizes large weights uniformly, preventing any single feature from dominating.",
    "Dropout randomly sets a fraction of neurons to zero during training, forcing the network to learn redundant representations and reducing co-adaptation.",
    "CNNs excel at image recognition, object detection, and any task involving spatial patterns due to their ability to capture local features hierarchically.",
    "CNNs use convolutional filters that slide across images, detecting features like edges, textures, and shapes at different scales through successive layers.",
    "Pooling reduces spatial dimensions by taking maximum or average values from regions, providing translation invariance and reducing computational cost.",
    "RNNs process sequential data by maintaining hidden state that captures information from previous time steps, useful for text, speech, and time series.",
    "LSTMs add gates that control information flow, solving the vanishing gradient problem and enabling learning of long-term dependencies in sequences.",
    "Attention allows models to focus on relevant parts of input when producing output, weighing the importance of different positions dynamically.",
    "Transformers use self-attention to process all positions simultaneously, enabling parallelization and capturing long-range dependencies more effectively than RNNs.",
    "GPT uses decoder-only architecture with causal masking, trained on next-token prediction at massive scale, enabling few-shot learning through in-context examples.",
    "Fine-tuning adapts a pre-trained model to specific tasks using domain data. Use it when you have limited labeled data but need task-specific performance.",
    "Use held-out test sets, cross-validation, and appropriate metrics for your task. Monitor for data leakage and ensure test data represents real-world distribution.",
    "For classification: accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix. Choice depends on class balance and error cost asymmetry.",
    "Precision measures what fraction of positive predictions are correct. Recall measures what fraction of actual positives are found. Trade-off depends on use case.",
    "F1 score is the harmonic mean of precision and recall, providing a balanced measure when you need to consider both false positives and false negatives.",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    """Metrics for a single API request."""

    payload_size_bytes: int = 0
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None  # Time to first token (streaming only)
    tokens_sent: int = 0
    tokens_received: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ConversationMetrics:
    """Aggregated metrics for a conversation benchmark."""

    mode: str  # "stateless" or "stateful"
    turns: int = 0
    total_payload_bytes: int = 0
    total_latency_ms: float = 0.0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    avg_payload_bytes: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_sent: float = 0.0
    request_metrics: List[RequestMetrics] = field(default_factory=list)

    def compute_averages(self) -> None:
        """Compute average metrics from individual request metrics."""
        if self.turns > 0:
            self.avg_payload_bytes = self.total_payload_bytes / self.turns
            self.avg_latency_ms = self.total_latency_ms / self.turns
            self.avg_tokens_sent = self.total_tokens_sent / self.turns


@dataclass
class StreamingMetrics:
    """Metrics specific to streaming comparison."""

    mode: str
    avg_ttft_ms: float = 0.0
    avg_total_ms: float = 0.0
    samples: int = 0


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a scenario."""

    scenario_name: str
    turns: int
    stateless: ConversationMetrics
    stateful: ConversationMetrics
    streaming_stateless: Optional[StreamingMetrics] = None
    streaming_stateful: Optional[StreamingMetrics] = None


# ---------------------------------------------------------------------------
# Token Estimation (Simple approximation)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a rough approximation; actual tokenization varies by model.
    """
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += estimate_tokens(item["text"])
        # Add overhead for role and structure
        total += 4
    return total


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class ResponsesAPIClient:
    """Client for the OpenAI Responses API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "test-key"
        self.model = model
        self.timeout = timeout
        self.tenant_id = tenant_id or f"bench-tenant-{uuid.uuid4().hex[:8]}"
        self.session_id = session_id or f"bench-session-{uuid.uuid4().hex[:8]}"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-tenant-id": self.tenant_id,
            "x-session-id": self.session_id,
        }

    async def create_response(
        self,
        input_data: Any,
        previous_response_id: Optional[str] = None,
        store: bool = True,
        stream: bool = False,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        """
        Send a request to POST /v1/responses.

        Returns the response data and request metrics.
        """
        metrics = RequestMetrics()

        # Build request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "store": store,
            "max_output_tokens": max_output_tokens,
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if stream:
            payload["stream"] = True

        # Calculate payload size and token estimate
        payload_json = json.dumps(payload)
        metrics.payload_size_bytes = len(payload_json.encode("utf-8"))

        # Estimate tokens sent
        if isinstance(input_data, str):
            metrics.tokens_sent = estimate_tokens(input_data)
        elif isinstance(input_data, list):
            metrics.tokens_sent = estimate_message_tokens(input_data)

        # Send request
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if stream:
                    response_data = {}
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/responses",
                        headers=self.headers,
                        content=payload_json,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if first_token_time is None and line.strip():
                                first_token_time = time.perf_counter()
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    event_data = json.loads(data_str)
                                    # Capture the final response
                                    if event_data.get("type") == "response.completed":
                                        response_data = event_data.get("response", {})
                                except json.JSONDecodeError:
                                    pass
                else:
                    response = await client.post(
                        f"{self.base_url}/responses",
                        headers=self.headers,
                        content=payload_json,
                    )
                    response.raise_for_status()
                    response_data = response.json()

            end_time = time.perf_counter()

            metrics.latency_ms = (end_time - start_time) * 1000
            if first_token_time:
                metrics.ttft_ms = (first_token_time - start_time) * 1000

            # Extract token usage if available
            usage = response_data.get("usage", {})
            if usage:
                metrics.tokens_sent = usage.get("input_tokens", metrics.tokens_sent)
                metrics.tokens_received = usage.get("output_tokens", 0)

            return response_data, metrics

        except httpx.HTTPStatusError as e:
            metrics.success = False
            metrics.error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            metrics.latency_ms = (time.perf_counter() - start_time) * 1000
            return {}, metrics

        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            metrics.latency_ms = (time.perf_counter() - start_time) * 1000
            return {}, metrics


# ---------------------------------------------------------------------------
# Dry Run Simulator
# ---------------------------------------------------------------------------


class DryRunSimulator:
    """Simulates API responses for dry-run mode."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.response_counter = 0

    def simulate_response(
        self,
        input_data: Any,
        previous_response_id: Optional[str] = None,
        store: bool = True,
        stream: bool = False,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        """Simulate an API response without making actual requests."""
        metrics = RequestMetrics()

        # Build simulated payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "store": store,
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if stream:
            payload["stream"] = True

        # Calculate payload metrics
        payload_json = json.dumps(payload)
        metrics.payload_size_bytes = len(payload_json.encode("utf-8"))

        # Estimate tokens sent
        if isinstance(input_data, str):
            metrics.tokens_sent = estimate_tokens(input_data)
        elif isinstance(input_data, list):
            metrics.tokens_sent = estimate_message_tokens(input_data)

        # Simulate latency (base + per-token generation)
        base_latency_ms = 50  # Base network latency
        token_latency_ms = 5  # Time per token generated

        # Get simulated response content
        response_idx = self.response_counter % len(SIMULATED_RESPONSES)
        response_content = SIMULATED_RESPONSES[response_idx]
        response_tokens = estimate_tokens(response_content)

        metrics.latency_ms = base_latency_ms + (response_tokens * token_latency_ms)
        metrics.tokens_received = response_tokens

        if stream:
            # Simulate TTFT (first token arrives quickly)
            metrics.ttft_ms = base_latency_ms + 10

        # Generate response ID
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.response_counter += 1

        # Build response object
        response_data = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": self.model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response_content}],
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                }
            ],
            "usage": {
                "input_tokens": metrics.tokens_sent,
                "output_tokens": metrics.tokens_received,
                "total_tokens": metrics.tokens_sent + metrics.tokens_received,
            },
        }

        return response_data, metrics


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs benchmark scenarios comparing stateful vs stateless approaches."""

    def __init__(
        self,
        client: Optional[ResponsesAPIClient] = None,
        simulator: Optional[DryRunSimulator] = None,
        dry_run: bool = False,
    ):
        self.client = client
        self.simulator = simulator or DryRunSimulator()
        self.dry_run = dry_run

    async def _send_request(
        self,
        input_data: Any,
        previous_response_id: Optional[str] = None,
        store: bool = True,
        stream: bool = False,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        """Send a request using either the real client or simulator."""
        if self.dry_run:
            return self.simulator.simulate_response(
                input_data, previous_response_id, store, stream
            )
        else:
            if not self.client:
                raise ValueError("Client required when not in dry-run mode")
            return await self.client.create_response(
                input_data, previous_response_id, store, stream
            )

    async def run_stateless_conversation(
        self, num_turns: int, stream: bool = False
    ) -> ConversationMetrics:
        """
        Run a conversation in stateless mode.

        Each request includes the full conversation history.
        """
        metrics = ConversationMetrics(mode="stateless")
        conversation_history: List[Dict[str, Any]] = []

        for turn_idx in range(num_turns):
            prompt_idx = turn_idx % len(CONVERSATION_PROMPTS)
            user_message = CONVERSATION_PROMPTS[prompt_idx]

            # Add user message to history
            conversation_history.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": user_message,
                }
            )

            # Send full history as input
            response_data, request_metrics = await self._send_request(
                input_data=conversation_history.copy(),
                previous_response_id=None,  # Stateless: no previous_response_id
                store=False,  # Stateless: no need to store
                stream=stream,
            )

            metrics.request_metrics.append(request_metrics)
            metrics.total_payload_bytes += request_metrics.payload_size_bytes
            metrics.total_latency_ms += request_metrics.latency_ms
            metrics.total_tokens_sent += request_metrics.tokens_sent
            metrics.total_tokens_received += request_metrics.tokens_received
            metrics.turns += 1

            # Extract assistant response and add to history
            output_items = response_data.get("output", [])
            for item in output_items:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    # Add assistant response to history for next turn
                    conversation_history.append(item)
                    break

        metrics.compute_averages()
        return metrics

    async def run_stateful_conversation(
        self, num_turns: int, stream: bool = False
    ) -> ConversationMetrics:
        """
        Run a conversation in stateful mode.

        Uses `previous_response_id` to chain responses, sending only new input.
        """
        metrics = ConversationMetrics(mode="stateful")
        previous_response_id: Optional[str] = None

        for turn_idx in range(num_turns):
            prompt_idx = turn_idx % len(CONVERSATION_PROMPTS)
            user_message = CONVERSATION_PROMPTS[prompt_idx]

            # Send only the new user message (string input for simplicity)
            response_data, request_metrics = await self._send_request(
                input_data=user_message,
                previous_response_id=previous_response_id,
                store=True,  # Stateful: store responses
                stream=stream,
            )

            metrics.request_metrics.append(request_metrics)
            metrics.total_payload_bytes += request_metrics.payload_size_bytes
            metrics.total_latency_ms += request_metrics.latency_ms
            metrics.total_tokens_sent += request_metrics.tokens_sent
            metrics.total_tokens_received += request_metrics.tokens_received
            metrics.turns += 1

            # Update previous_response_id for next turn
            previous_response_id = response_data.get("id")

        metrics.compute_averages()
        return metrics

    async def run_streaming_comparison(
        self, num_samples: int = 3
    ) -> Tuple[StreamingMetrics, StreamingMetrics]:
        """
        Compare streaming performance for stateless vs stateful.

        Measures time-to-first-token (TTFT) and total completion time.
        """
        stateless_metrics = StreamingMetrics(mode="stateless", samples=num_samples)
        stateful_metrics = StreamingMetrics(mode="stateful", samples=num_samples)

        stateless_ttfts: List[float] = []
        stateless_totals: List[float] = []
        stateful_ttfts: List[float] = []
        stateful_totals: List[float] = []

        for _ in range(num_samples):
            # Stateless streaming
            conversation = [
                {"type": "message", "role": "user", "content": CONVERSATION_PROMPTS[0]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": SIMULATED_RESPONSES[0],
                },
                {"type": "message", "role": "user", "content": CONVERSATION_PROMPTS[1]},
            ]
            _, metrics = await self._send_request(
                input_data=conversation, store=False, stream=True
            )
            if metrics.ttft_ms:
                stateless_ttfts.append(metrics.ttft_ms)
            stateless_totals.append(metrics.latency_ms)

            # Stateful streaming (simulate with stored response)
            # First, create the initial response
            resp1, _ = await self._send_request(
                input_data=CONVERSATION_PROMPTS[0], store=True, stream=False
            )
            prev_id = resp1.get("id")

            # Then stream the second turn
            _, metrics = await self._send_request(
                input_data=CONVERSATION_PROMPTS[1],
                previous_response_id=prev_id,
                store=True,
                stream=True,
            )
            if metrics.ttft_ms:
                stateful_ttfts.append(metrics.ttft_ms)
            stateful_totals.append(metrics.latency_ms)

        if stateless_ttfts:
            stateless_metrics.avg_ttft_ms = sum(stateless_ttfts) / len(stateless_ttfts)
        stateless_metrics.avg_total_ms = sum(stateless_totals) / len(stateless_totals)

        if stateful_ttfts:
            stateful_metrics.avg_ttft_ms = sum(stateful_ttfts) / len(stateful_ttfts)
        stateful_metrics.avg_total_ms = sum(stateful_totals) / len(stateful_totals)

        return stateless_metrics, stateful_metrics

    async def run_scenario(
        self, name: str, num_turns: int, include_streaming: bool = True
    ) -> BenchmarkResult:
        """Run a complete benchmark scenario."""
        print(f"\nRunning scenario: {name} ({num_turns} turns)")
        print("-" * 50)

        # Run non-streaming benchmarks
        print("  Running stateless benchmark...")
        stateless = await self.run_stateless_conversation(num_turns, stream=False)

        print("  Running stateful benchmark...")
        stateful = await self.run_stateful_conversation(num_turns, stream=False)

        result = BenchmarkResult(
            scenario_name=name,
            turns=num_turns,
            stateless=stateless,
            stateful=stateful,
        )

        # Run streaming comparison if requested
        if include_streaming:
            print("  Running streaming comparison...")
            stream_stateless, stream_stateful = await self.run_streaming_comparison()
            result.streaming_stateless = stream_stateless
            result.streaming_stateful = stream_stateful

        return result


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def format_bytes(num_bytes: float) -> str:
    """Format bytes as human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


def format_number(num: float) -> str:
    """Format number with thousand separators."""
    if num >= 1000:
        return f"{num:,.0f}"
    return f"{num:.0f}"


def calculate_improvement(baseline: float, improved: float) -> str:
    """Calculate percentage improvement."""
    if baseline == 0:
        return "N/A"
    improvement = ((baseline - improved) / baseline) * 100
    if improvement > 0:
        return f"{improvement:.1f}%"
    else:
        return f"{-improvement:.1f}% slower"


def print_results_table(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted table."""
    stateless = result.stateless
    stateful = result.stateful

    # Calculate improvements
    payload_improvement = calculate_improvement(
        stateless.total_payload_bytes, stateful.total_payload_bytes
    )
    latency_improvement = calculate_improvement(
        stateless.avg_latency_ms, stateful.avg_latency_ms
    )
    tokens_improvement = calculate_improvement(
        stateless.total_tokens_sent, stateful.total_tokens_sent
    )

    print(f"\nScenario: {result.scenario_name}")
    print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 13 + "+")
    print(
        f"| {'Mode':<11} | {'Payload Size':<12} | {'Avg Latency':<12} | {'Tokens Sent':<11} |"
    )
    print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 13 + "+")
    print(
        f"| {'Stateless':<11} | {format_bytes(stateless.total_payload_bytes):<12} | {stateless.avg_latency_ms:>9.0f}ms | {format_number(stateless.total_tokens_sent):>11} |"
    )
    print(
        f"| {'Stateful':<11} | {format_bytes(stateful.total_payload_bytes):<12} | {stateful.avg_latency_ms:>9.0f}ms | {format_number(stateful.total_tokens_sent):>11} |"
    )
    print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 13 + "+")
    print(
        f"| {'Improvement':<11} | {payload_improvement:<12} | {latency_improvement:<12} | {tokens_improvement:<11} |"
    )
    print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 13 + "+")

    # Streaming results if available
    if result.streaming_stateless and result.streaming_stateful:
        stream_sl = result.streaming_stateless
        stream_sf = result.streaming_stateful

        ttft_improvement = calculate_improvement(
            stream_sl.avg_ttft_ms, stream_sf.avg_ttft_ms
        )
        total_improvement = calculate_improvement(
            stream_sl.avg_total_ms, stream_sf.avg_total_ms
        )

        print("\nStreaming Comparison:")
        print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+")
        print(f"| {'Mode':<11} | {'Avg TTFT':<12} | {'Avg Total':<12} |")
        print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+")
        print(
            f"| {'Stateless':<11} | {stream_sl.avg_ttft_ms:>9.0f}ms | {stream_sl.avg_total_ms:>9.0f}ms |"
        )
        print(
            f"| {'Stateful':<11} | {stream_sf.avg_ttft_ms:>9.0f}ms | {stream_sf.avg_total_ms:>9.0f}ms |"
        )
        print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+")
        print(
            f"| {'Improvement':<11} | {ttft_improvement:<12} | {total_improvement:<12} |"
        )
        print("+" + "-" * 13 + "+" + "-" * 14 + "+" + "-" * 14 + "+")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print overall summary of all scenarios."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for result in results:
        stateless = result.stateless
        stateful = result.stateful

        payload_reduction = (
            (1 - stateful.total_payload_bytes / stateless.total_payload_bytes) * 100
            if stateless.total_payload_bytes > 0
            else 0
        )
        token_reduction = (
            (1 - stateful.total_tokens_sent / stateless.total_tokens_sent) * 100
            if stateless.total_tokens_sent > 0
            else 0
        )

        print(f"\n{result.scenario_name}:")
        print(f"  Payload reduction: {payload_reduction:.1f}%")
        print(f"  Token reduction:   {token_reduction:.1f}%")
        print(
            f"  Total stateless payload: {format_bytes(stateless.total_payload_bytes)}"
        )
        print(f"  Total stateful payload:  {format_bytes(stateful.total_payload_bytes)}")

    print("\n" + "=" * 70)
    print(
        "Note: Stateful mode is most beneficial for longer conversations where"
    )
    print("      payload and token savings compound with each additional turn.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark stateful vs stateless Responses API usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dry-run mode (no server required)
  python benchmark_stateful_responses.py --dry-run

  # Run against local server
  python benchmark_stateful_responses.py --base-url http://localhost:9000/v1

  # Custom model and turn count
  python benchmark_stateful_responses.py --base-url http://localhost:9000/v1 --model gpt-4 --turns 15

  # Output JSON results
  python benchmark_stateful_responses.py --dry-run --json results.json
        """,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:9000/v1",
        help="Base URL for the Responses API (default: http://localhost:9000/v1)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: test-key)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for benchmarks (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Run a single scenario with this many turns (default: run 3, 10, 25 turn scenarios)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate API calls without hitting server (for testing)",
    )

    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Skip streaming comparison tests",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )

    parser.add_argument(
        "--tenant-id",
        type=str,
        default=None,
        help="Tenant ID for session isolation (auto-generated if not provided)",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for session isolation (auto-generated if not provided)",
    )

    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="FILE",
        help="Output results to JSON file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-request metrics",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    print("=" * 70)
    print("Responses API Benchmark: Stateful vs Stateless")
    print("=" * 70)

    if args.dry_run:
        print("\nMode: DRY RUN (simulated responses)")
        print("No actual API calls will be made.")
    else:
        print(f"\nMode: LIVE")
        print(f"Base URL: {args.base_url}")
        print(f"Model: {args.model}")

    # Initialize client/simulator
    client = None
    if not args.dry_run:
        client = ResponsesAPIClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            tenant_id=args.tenant_id,
            session_id=args.session_id,
        )
        print(f"Tenant ID: {client.tenant_id}")
        print(f"Session ID: {client.session_id}")

    simulator = DryRunSimulator(model=args.model)
    runner = BenchmarkRunner(client=client, simulator=simulator, dry_run=args.dry_run)

    # Define scenarios
    if args.turns:
        scenarios = [(f"{args.turns}-turn conversation", args.turns)]
    else:
        scenarios = [
            ("Short conversation (3 turns)", 3),
            ("Medium conversation (10 turns)", 10),
            ("Long conversation (25 turns)", 25),
        ]

    # Run benchmarks
    results: List[BenchmarkResult] = []
    for name, turns in scenarios:
        try:
            result = await runner.run_scenario(
                name=name,
                num_turns=turns,
                include_streaming=not args.no_streaming,
            )
            results.append(result)
            print_results_table(result)

            if args.verbose:
                print("\nPer-request metrics (Stateless):")
                for i, m in enumerate(result.stateless.request_metrics):
                    print(
                        f"  Turn {i + 1}: {format_bytes(m.payload_size_bytes)} payload, "
                        f"{m.latency_ms:.0f}ms latency, {m.tokens_sent} tokens"
                    )
                print("\nPer-request metrics (Stateful):")
                for i, m in enumerate(result.stateful.request_metrics):
                    print(
                        f"  Turn {i + 1}: {format_bytes(m.payload_size_bytes)} payload, "
                        f"{m.latency_ms:.0f}ms latency, {m.tokens_sent} tokens"
                    )

        except Exception as e:
            print(f"\nError running scenario '{name}': {e}")
            if not args.dry_run:
                print("Hint: Use --dry-run to test without a server")
            return 1

    # Print summary
    print_summary(results)

    # Export JSON if requested
    if args.json:
        export_data = {
            "metadata": {
                "base_url": args.base_url,
                "model": args.model,
                "dry_run": args.dry_run,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "results": [],
        }
        for result in results:
            export_data["results"].append(
                {
                    "scenario": result.scenario_name,
                    "turns": result.turns,
                    "stateless": {
                        "total_payload_bytes": result.stateless.total_payload_bytes,
                        "avg_latency_ms": result.stateless.avg_latency_ms,
                        "total_tokens_sent": result.stateless.total_tokens_sent,
                        "total_tokens_received": result.stateless.total_tokens_received,
                    },
                    "stateful": {
                        "total_payload_bytes": result.stateful.total_payload_bytes,
                        "avg_latency_ms": result.stateful.avg_latency_ms,
                        "total_tokens_sent": result.stateful.total_tokens_sent,
                        "total_tokens_received": result.stateful.total_tokens_received,
                    },
                }
            )

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\nResults exported to: {args.json}")

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

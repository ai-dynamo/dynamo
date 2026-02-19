<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Async GRPO Rollout Example

This directory contains a simplified demonstration of asynchronous Group Relative Policy Optimization (GRPO) using any OpenAI-compatible inference backend (e.g., Dynamo SGLang workers).

## Overview

Async GRPO enables continuous trajectory generation concurrent with model training, improving throughput by overlapping generation and training phases.
This implementation demonstrates the core concepts while using mock training to focus on the async collection workflow.

### What This Example Demonstrates
- Core async GRPO workflow (concurrent generation + training)
- Replay buffer mechanics with weight versioning
- Background trajectory collection via OpenAI-compatible HTTP API
- Multi-turn conversation support
- Target weight calculation logic
- Pause/resume for weight updates

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Training Loop                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Sample trajectories from replay buffer              │ │
│  │ 2. Train policy on sampled batch                       │ │
│  │ 3. Update weight version                               │ │
│  │ 4. Notify collector to continue with new weights       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕ (async)
┌─────────────────────────────────────────────────────────────┐
│              Background Trajectory Collector                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Generate trajectories via OpenAI-compatible API     │ │
│  │ 2. Compute rewards                                     │ │
│  │ 3. Push to replay buffer with version metadata        │ │
│  │ 4. Continue generating for future training steps      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Async GRPO Training Loop (`async_grpo.py`)

The main entry point demonstrating the async GRPO workflow:

- **Concurrent Generation + Training**: Trajectories are generated in the background while training proceeds
- **Weight Versioning**: Tracks which model weights were used for generation vs. which training step they target
- **Replay Buffer Management**: Coordinates sampling and buffer filling

**Key Features:**
- Uses `asyncio` with `uvloop` for high-performance async operations
- Connects to any OpenAI-compatible API endpoint via `AsyncOpenAI`
- Demonstrates pause/resume during weight updates
- Per-step throughput and timing metrics

### 2. Shared Data Types (`utils.py`)

Dataclasses used across all components:

- **`Config`**: All configuration parameters (API, dataset, GRPO, generation, async settings)
- **`Turn`**: A single conversation turn with user input, generated output, reward, logprobs, and timing
- **`Trajectory`**: A complete conversation with weight version metadata for importance sampling

### 3. Trajectory Collector (`components/trajectory.py`)

Manages asynchronous trajectory generation via HTTP:

```python
class HttpTrajectoryCollector:
    """Collects trajectories asynchronously via HTTP (OpenAI-compatible API)."""
```

**Key Features:**
- **Multi-Turn Generation**: Iterates through conversation turns, building message history
- **Target Weight Calculation**: Determines which future training steps can use trajectories from current weights
- **Concurrency Control**: Limits parallel requests via `asyncio.Semaphore`
- **Streaming Task Pool**: Creates tasks incrementally (not all at once) to prevent OOM
- **Pause/Resume Events**: Separate events for manual pause and weight refit coordination
- **Logprob Extraction**: Captures per-token logprobs from the API for importance sampling

**Weight Versioning Logic:**
- Generation weight: Model version used to generate trajectory
- Target weight: Training step where trajectory will be used
- Max trajectory age: How many steps old trajectories can be

Example:
```
Current weight version: 10
Max trajectory age: 3 steps

Generates trajectories for target weights: [11, 12, 13]
```

### 4. Replay Buffer (`components/replay_buffer.py`)

Thread-safe buffer for storing trajectories with metadata:

```python
class ReplayBuffer:
    """Replay buffer with age-based trajectory management."""
```

**Key Features:**
- **Version Tracking**: Stores generation and target weight versions
- **Age Filtering**: Expires trajectories older than `max_trajectory_age_steps`
- **Backpressure**: Returns "full" status when at capacity; collector retries with backoff
- **Statistics**: Tracks pushed, sampled, and expired trajectories

## Running the Example

### Prerequisites

1. An OpenAI-compatible inference server running (e.g., Dynamo SGLang workers)
2. If using Dynamo: ETCD and NATS running for the Dynamo runtime
3. Python dependencies: `openai`, `uvloop`, `datasets` (for HuggingFace data)

### Launch Scripts

Launch scripts for Dynamo SGLang workers are available in `launch/`:

**Disaggregated (prefill/decode split):**
```bash
./launch/sglang_disagg.sh 4   # 3 decode + 1 prefill
./launch/sglang_disagg.sh     # Auto-detect (N-1) decode + 1 prefill
```

**Aggregated:**
```bash
./launch/sglang_agg.sh 4      # 4 monolithic workers
./launch/sglang_agg.sh         # Auto-detect GPUs
```

### Basic Usage

```bash
# 1. Start Dynamo SGLang workers (disaggregated)
./launch/sglang_disagg.sh 4

# 2. Run async GRPO training
python async_grpo.py \
    --api-base http://localhost:8000/v1 \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num-prompts-per-step 128 \
    --num-generations-per-prompt 8 \
    --num-steps 10 \
    --max-trajectory-age 1
```

### Configuration Options

**API Settings:**
- `--api-base`: OpenAI-compatible API base URL (default: `http://localhost:8000/v1`)
- `--model-name`: Model name for generation (default: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)

**Dataset:**
- `--dataset-path`: Path to local JSONL dataset file
- `--dataset-name`: HuggingFace dataset name (default: `nvidia/Nemotron-Instruction-Following-Chat-v1`)
- `--dataset-subset`: HuggingFace dataset subset (default: `chat_if`)

**GRPO Parameters:**
- `--num-prompts-per-step`: Number of prompts per training step (default: 128)
- `--num-generations-per-prompt`: Generations per prompt (default: 8)
- `--num-steps`: Total training steps (default: 1)
- `--max-trajectory-age`: Max age in steps for trajectories (default: 1)

**Generation Parameters:**
- `--max-new-tokens`: Max tokens to generate (default: 8192)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max-concurrency`: Max concurrent HTTP requests (default: 1024)

## References

- **NeMo-RL Async GRPO**: Production implementation with all features
  - `NeMo-RL/nemo_rl/algorithms/grpo.py:async_grpo_train`
  - `NeMo-RL/nemo_rl/algorithms/async_utils.py`
- **Dynamo Runtime Documentation**: See `dynamo/` root directory

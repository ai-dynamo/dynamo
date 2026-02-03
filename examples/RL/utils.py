import time
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Config:
    # Model API
    api_base: str = "http://localhost:8000/v1"
    namespace: str = "dynamo"
    component: str = "backend"
    endpoint: str = "generate"
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # Dataset
    dataset_name: str = "nvidia/Nemotron-Instruction-Following-Chat-v1"
    dataset_subset: str = "chat_if"
    dataset_path: str = ""

    # GRPO
    num_steps: int = 1
    num_prompts_per_step: int = 128
    num_generations_per_prompt: int = 8
    max_trajectory_age_steps: int = 1

    # Generation params
    max_new_tokens: int = 8192
    temperature: float = 1.0
    max_concurrency: int = 1024
    # partial_rollout_enabled: bool = False

    # Async settings
    buffer_wait_timeout: float = 1000.0  # Max seconds to wait for buffer fill

    # Runtime settings
    store_kv: str = "etcd"
    request_plane: str = "tcp"


@dataclass
class Turn:
    """A single turn in a conversation."""

    user: str
    ground_truth: str = ""
    generated: str = ""
    reward: float = 0.0
    logprobs: List[Any] = field(default_factory=list)
    # Per-token log probabilities for importance sampling
    token_logprobs: List[float] = field(default_factory=list)
    # Timing metrics
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trajectory:
    """A complete trajectory (conversation) with metadata."""

    id: int
    gen_idx: int  # Which generation (0 to num_generations_per_prompt-1)
    turns: List[Turn]
    total_reward: float = 0.0
    # Weight version tracking for importance sampling
    generation_weight_version: int = 0
    target_weight_version: int = 0
    timestamp: float = field(default_factory=time.time)

    def get_generation_logprobs(self) -> List[float]:
        """Get all token logprobs from all turns for importance sampling."""
        all_logprobs = []
        for turn in self.turns:
            all_logprobs.extend(turn.token_logprobs)
        return all_logprobs

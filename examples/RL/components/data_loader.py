import json
import logging
import os
from typing import Dict, List

from utils import Config, Turn

logger = logging.getLogger(__name__)


def load_data(cfg: Config) -> List[Dict]:
    """Load data from JSONL or HuggingFace dataset."""
    if cfg.dataset_path:
        return _load_jsonl(cfg)
    elif cfg.dataset_name:
        return _load_huggingface(cfg)
    else:
        raise ValueError("Must specify either --dataset-path or --dataset-name")


def _load_huggingface(cfg: Config) -> List[Dict]:
    """Load from HuggingFace dataset."""
    from datasets import load_dataset

    logger.info(f"Loading dataset {cfg.dataset_name}")

    if cfg.dataset_subset:
        dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_subset)
    else:
        dataset = load_dataset(cfg.dataset_name, split="train")

    data = []
    for idx, entry in enumerate(dataset):
        messages = entry.get("messages", [])
        if not messages:
            continue

        turns = []
        pending_user = None

        for msg in messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                turns.append(Turn(user=pending_user, ground_truth=content))
                pending_user = None

        if turns:
            data.append({"id": idx, "turns": turns})

    logger.info(f"  Loaded {len(data)} conversations")
    return data


def _load_jsonl(cfg: Config) -> List[Dict]:
    """Load from local JSONL file."""
    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {cfg.dataset_path}")

    logger.info(f"Loading dataset from {cfg.dataset_path}")
    data = []

    with open(cfg.dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            prompt_text = item.get("prompt", "")
            ground_truth = item.get("answer", item.get("ground_truth", ""))

            turn = Turn(user=prompt_text, ground_truth=ground_truth)
            data.append({"id": idx, "turns": [turn]})

    logger.info(f"  Loaded {len(data)} prompts")
    return data

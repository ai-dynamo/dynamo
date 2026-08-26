#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove ProRL trace samples were consumed by one valid Miles NVFP4 step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--driver-log", required=True)
    parser.add_argument("--evidence-dir", required=True)
    return parser.parse_args()


def as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def token_hash(tokens: Any) -> str:
    payload = ",".join(str(token) for token in as_int_list(tokens)).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    dump = run_dir / "dump_details"
    rollout_dump = torch.load(
        dump / "rollout_data" / "0.pt", map_location="cpu", weights_only=False
    )
    train_dump = torch.load(
        dump / "train_data" / "0_0.pt", map_location="cpu", weights_only=False
    )
    raw_samples = rollout_dump["samples"]
    train = train_dump["rollout_data"]

    assert len(raw_samples) == 3, (
        "Agent execution must yield root, child, and resumed-root traces"
    )
    raw_hashes: list[str] = []
    raw_weight_versions: list[tuple[str, ...]] = []
    raw_session_ids: set[str] = set()
    raw_rollout_ids: set[int] = set()
    raw_summaries: list[dict[str, Any]] = []
    responses: list[str] = []
    samples_by_trace: dict[int, dict[str, Any]] = {}
    for sample in raw_samples:
        tokens = as_int_list(sample["tokens"])
        response_length = int(sample["response_length"])
        loss_mask = as_int_list(sample["loss_mask"])
        logprobs = [float(value) for value in sample["rollout_log_probs"]]
        assert response_length > 0
        assert len(loss_mask) == response_length
        assert sum(loss_mask) > 0
        assert len(logprobs) == response_length
        assert all(math.isfinite(value) for value in logprobs)
        assert float(sample["reward"]["score"]) == 1.0
        assert sample["weight_versions"], sample["weight_versions"]
        raw_weight_versions.append(
            tuple(str(value) for value in sample["weight_versions"])
        )

        routed = np.asarray(sample["rollout_routed_experts"])
        assert routed.shape == (len(tokens) - 1, 48, 8), routed.shape
        assert routed.dtype == np.int32
        assert int(routed.min()) >= 0
        assert int(routed.max()) < 128

        polar = sample["metadata"]["polar"]
        trace_index = int(polar["trace_index"])
        assert trace_index not in samples_by_trace, trace_index
        samples_by_trace[trace_index] = sample
        raw_session_ids.add(str(polar["session_id"]))
        raw_rollout_ids.add(int(sample["rollout_id"]))
        responses.append(str(sample["response"]))
        digest = token_hash(tokens)
        raw_hashes.append(digest)
        raw_summaries.append(
            {
                "sample_index": int(sample["index"]),
                "rollout_id": int(sample["rollout_id"]),
                "trace_index": trace_index,
                "session_id": str(polar["session_id"]),
                "task_id": str(polar["task_id"]),
                "token_count": len(tokens),
                "response_length": response_length,
                "active_loss_tokens": sum(loss_mask),
                "token_sha256": digest,
                "weight_versions": [str(value) for value in sample["weight_versions"]],
                "routed_experts_shape": list(routed.shape),
            }
        )

    assert len(raw_session_ids) == 1, raw_session_ids
    assert len(raw_rollout_ids) == 1, raw_rollout_ids
    assert sorted(samples_by_trace) == [0, 1, 2], sorted(samples_by_trace)
    root = samples_by_trace[0]
    child = samples_by_trace[1]
    resumed_root = samples_by_trace[2]
    root_debug = root["metadata"]["polar"]["trace_debug"]
    assert root_debug["finish_reason"] == "tool_calls", root_debug
    agent_calls = [
        call
        for message in root_debug["response_messages"]
        for call in message.get("tool_calls", [])
        if call.get("function", {}).get("name") == "Agent"
    ]
    assert len(agent_calls) == 1, agent_calls
    agent_call = agent_calls[0]
    assert json.loads(agent_call["function"]["arguments"]) == {
        "description": "Calculate product",
        "prompt": "Calculate 13 * 17 and return exactly CHILD_221.",
        "subagent_type": "general-purpose",
    }, agent_call
    child_prompt = json.dumps(child["prompt"], ensure_ascii=False)
    assert "You are an agent for Claude Code" in child_prompt, child_prompt
    assert "Calculate 13 * 17 and return exactly CHILD_221." in child_prompt
    assert str(child["response"]).rstrip().endswith("CHILD_221"), child["response"]
    resumed_prompt = json.dumps(resumed_root["prompt"], ensure_ascii=False)
    assert json.dumps({"role": "tool"})[1:-1] in resumed_prompt, resumed_prompt
    assert agent_call["id"] in resumed_prompt, resumed_prompt
    assert "CHILD_221" in resumed_prompt and "agentId:" in resumed_prompt
    assert str(resumed_root["response"]).rstrip().endswith("ROOT_OK CHILD_221"), (
        resumed_root["response"]
    )
    joined_responses = "\n".join(responses)
    assert "CHILD_221" in joined_responses
    assert "ROOT_OK CHILD_221" in joined_responses

    train_hashes = [token_hash(tokens) for tokens in train["tokens"]]
    assert Counter(train_hashes) == Counter(raw_hashes), (train_hashes, raw_hashes)
    train_weight_versions = [
        tuple(str(value) for value in versions)
        for versions in train["weight_versions"]
    ]
    train_token_versions = list(zip(train_hashes, train_weight_versions, strict=True))
    raw_token_versions = list(zip(raw_hashes, raw_weight_versions, strict=True))
    assert Counter(train_token_versions) == Counter(raw_token_versions), (
        train_token_versions,
        raw_token_versions,
    )
    assert len(train["sample_indices"]) == len(raw_samples)
    assert set(int(value) for value in train["rollout_ids"]) == raw_rollout_ids
    assert all(float(value) > 0 for value in train["raw_reward"]), train["raw_reward"]

    advantages = [
        tensor.detach().cpu().float()
        for tensor in train["advantages"]
    ]
    assert advantages
    assert all(torch.isfinite(tensor).all().item() for tensor in advantages)
    assert any(torch.count_nonzero(tensor).item() > 0 for tensor in advantages)
    assert all(float(mask_sum) > 0 for mask_sum in train["rollout_mask_sums"])

    event_path = dump / "events" / "actor_cell0_rank0.jsonl"
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    train_metrics = [
        event["metrics"]
        for event in events
        if isinstance(event.get("metrics"), dict)
        and "train/grad_norm" in event["metrics"]
    ]
    assert train_metrics, events
    grad_norm = float(train_metrics[-1]["train/grad_norm"])
    assert math.isfinite(grad_norm) and grad_norm > 0, grad_norm

    driver_log = Path(cli.driver_log).read_text(errors="replace")
    valid_marker = (
        "op=train_step rollout=0 step=0 attempt=0 outcome=NORMAL valid_step=true"
    )
    assert valid_marker in driver_log
    assert re.search(
        r"use_rollout_routing_replay\s+\.+\s+True", driver_log
    ), "Miles did not log rollout routing replay as enabled"
    expected_config = {
        "fp4_recipe": "nvfp4",
        "fp4": "e2m1",
        "transformer_impl": "transformer_engine",
        "hf_checkpoint": "/opt/models/Qwen3-30B-A3B-NVFP4/",
    }
    for name, value in expected_config.items():
        assert re.search(rf"{name}\s+\.+\s+{re.escape(value)}", driver_log), (
            name,
            value,
        )
    assert re.search(r"te_precision_config_file\s+\.+\s+base64:", driver_log)
    assert "rollout_routed_experts" not in train, (
        "fill_replay_data must consume and delete routed experts before the saved "
        "train dump"
    )
    valid_position = driver_log.index(valid_marker)
    rank0_sync_positions = [
        match.start()
        for match in re.finditer(
            r"actor_cell0_rank0\].*fn=update_weights phase=end ok=true",
            driver_log,
        )
    ]
    assert any(position < valid_position for position in rank0_sync_positions), (
        valid_position,
        rank0_sync_positions,
    )
    assert any(position > valid_position for position in rank0_sync_positions), (
        valid_position,
        rank0_sync_positions,
    )
    assert "successfully saved checkpoint from iteration" in driver_log
    checkpoint = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    assert checkpoint.is_file(), checkpoint

    evidence_dir = Path(cli.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "result": "passed",
        "boundary": (
            "ProRL agent/subagent samples joined exactly to Miles optimizer input"
        ),
        "sample_count": len(raw_samples),
        "session_ids": sorted(raw_session_ids),
        "rollout_ids": sorted(raw_rollout_ids),
        "raw_samples": raw_summaries,
        "train_token_sha256": train_hashes,
        "raw_weight_versions": [list(values) for values in raw_weight_versions],
        "train_weight_versions": [list(values) for values in train_weight_versions],
        "token_weight_version_pairs": [
            [digest, list(versions)] for digest, versions in train_token_versions
        ],
        "sample_indices": [int(value) for value in train["sample_indices"]],
        "train_rollout_ids": [int(value) for value in train["rollout_ids"]],
        "raw_reward": [float(value) for value in train["raw_reward"]],
        "advantages_nonzero": [
            int(torch.count_nonzero(tensor).item()) for tensor in advantages
        ],
        "gradient_norm": grad_norm,
        "valid_step": True,
        "routing_replay_consumed": True,
        "nvfp4_training_config": expected_config,
        "post_step_sync": True,
        "checkpoint_iteration": checkpoint.read_text().strip(),
    }
    (evidence_dir / "train-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

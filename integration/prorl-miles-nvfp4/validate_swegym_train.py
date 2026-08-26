#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove real ProRL SWE-Gym traces drove one valid Miles NVFP4 step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swegym_fidelity import (
    load_and_match_trace_records,
    load_trace_snapshot,
    tool_names,
    validate_session_fidelity,
)

INSTANCE_ID = "getmoto__moto-7365"
EXPECTED_SESSIONS = 4
EXPECTED_TRAIN_RANKS = 4
EXPECTED_CHECKPOINT_ITERATION = "0"
MAX_SEQUENCE_LENGTH = 131072
MAX_RESPONSE_LENGTH = 4096
EXPECTED_PATCH_SHA256 = "bd25ef8a5b93198b2eccb99d9ed8bcbc2275ebfba50695df386a26f40fe8b6ce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--driver-log", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--attempt-start-file", required=True)
    parser.add_argument("--pre-server-info", required=True)
    parser.add_argument("--server-info-url", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--trace-start-file", required=True)
    return parser.parse_args()


def as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def token_hash(tokens: Any) -> str:
    payload = ",".join(str(token) for token in as_int_list(tokens)).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluation_for(sample: dict[str, Any]) -> dict[str, Any]:
    polar = sample["metadata"]["polar"]
    evaluation = polar["trajectory_metadata"]["evaluation"]
    assert evaluation["strategy"] == "swebench_harness", evaluation
    report = evaluation["report"]
    for flag in ("empty_generation", "failed_apply_patch", "error_eval", "test_timeout"):
        assert report[flag] is False, report
    patch = evaluation["patch"]
    assert isinstance(patch, str) and patch.strip(), evaluation
    assert evaluation["patch_sha256"] == hashlib.sha256(patch.encode()).hexdigest()
    assert evaluation["patch_sha256"] == EXPECTED_PATCH_SHA256, evaluation["patch_sha256"]
    assert "__POLAR_APPLY_PATCH_PASS__" in evaluation["apply_patch_output"]
    assert report["grading_report"], report
    reward = float(sample["reward"]["score"])
    assert reward in (0.0, 1.0), reward
    assert report["resolved"] is True, report
    assert reward == 1.0, reward
    return evaluation


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    attempt_start_file = Path(cli.attempt_start_file)
    assert attempt_start_file.is_file(), attempt_start_file
    attempt_start_ns = int(attempt_start_file.read_text().strip())

    def require_fresh(path: Path) -> Path:
        assert path.is_file(), path
        assert path.stat().st_mtime_ns >= attempt_start_ns, (
            path,
            path.stat().st_mtime_ns,
            attempt_start_ns,
        )
        return path

    dump = run_dir / "dump_details"
    rollout_path = require_fresh(dump / "rollout_data" / "0.pt")
    train_path = require_fresh(dump / "train_data" / "0_0.pt")
    rollout_dump = torch.load(
        rollout_path, map_location="cpu", weights_only=False
    )
    train_dump = torch.load(
        train_path, map_location="cpu", weights_only=False
    )
    raw_samples = rollout_dump["samples"]
    train = train_dump["rollout_data"]
    assert len(raw_samples) == EXPECTED_SESSIONS * 4, len(raw_samples)

    raw_hashes: list[str] = []
    raw_weight_versions: list[tuple[str, ...]] = []
    raw_session_ids: set[str] = set()
    raw_rollout_ids: set[int] = set()
    raw_identities: list[tuple[int, int, str, tuple[str, ...]]] = []
    raw_summaries: list[dict[str, Any]] = []
    session_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_identities: dict[str, set[tuple[int, int]]] = defaultdict(set)
    evaluations: dict[str, dict[str, Any]] = {}
    for sample in raw_samples:
        tokens = as_int_list(sample["tokens"])
        assert len(tokens) <= MAX_SEQUENCE_LENGTH, len(tokens)
        response_length = int(sample["response_length"])
        loss_mask = as_int_list(sample["loss_mask"])
        logprobs = [float(value) for value in sample["rollout_log_probs"]]
        assert response_length > 0
        assert response_length <= MAX_RESPONSE_LENGTH, response_length
        assert sample["status"] == "completed", sample["status"]
        assert sample["remove_sample"] is False
        assert len(loss_mask) == response_length
        assert sum(loss_mask) > 0
        assert len(logprobs) == response_length
        assert all(math.isfinite(value) for value in logprobs)
        versions = tuple(str(value) for value in sample["weight_versions"])
        assert versions == ("1",), versions
        raw_weight_versions.append(versions)

        routed = np.asarray(sample["rollout_routed_experts"])
        assert routed.shape == (len(tokens) - 1, 48, 8), routed.shape
        assert routed.dtype == np.int32
        assert int(routed.min()) >= 0 and int(routed.max()) < 128

        polar = sample["metadata"]["polar"]
        assert polar["result_metadata"]["instance_id"] == INSTANCE_ID
        session_id = str(polar["session_id"])
        session_samples[session_id].append(sample)
        sample_index = int(sample["index"])
        rollout_id = int(sample["rollout_id"])
        session_identities[session_id].add((sample_index, rollout_id))
        evaluations[session_id] = evaluation_for(sample)
        raw_session_ids.add(session_id)
        raw_rollout_ids.add(rollout_id)
        digest = token_hash(tokens)
        raw_hashes.append(digest)
        raw_identities.append((sample_index, rollout_id, digest, versions))
        raw_summaries.append(
            {
                "sample_index": sample_index,
                "rollout_id": rollout_id,
                "trace_index": int(polar["trace_index"]),
                "session_id": session_id,
                "task_id": str(polar["task_id"]),
                "finish_reason": polar["trace_debug"]["finish_reason"],
                "tool_names": tool_names(sample),
                "reward": float(sample["reward"]["score"]),
                "token_count": len(tokens),
                "response_length": response_length,
                "active_loss_tokens": sum(loss_mask),
                "token_sha256": digest,
                "weight_versions": [str(value) for value in sample["weight_versions"]],
                "routed_experts_shape": list(routed.shape),
            }
        )

    assert len(raw_session_ids) == EXPECTED_SESSIONS, raw_session_ids
    assert len(raw_rollout_ids) == EXPECTED_SESSIONS, raw_rollout_ids
    assert all(len(identities) == 1 for identities in session_identities.values()), (
        session_identities
    )
    session_identity = {
        session_id: next(iter(identities))
        for session_id, identities in session_identities.items()
    }
    assert len(set(session_identity.values())) == EXPECTED_SESSIONS, session_identity
    trace_start = load_trace_snapshot(Path(cli.trace_start_file))
    contexts, matched_trace_records, trace_records = load_and_match_trace_records(
        Path(cli.trace_dir), trace_start, raw_samples
    )
    session_rewards: dict[str, float] = {}
    session_trace_counts: dict[str, int] = {}
    session_tool_names: dict[str, list[str]] = {}
    session_fidelity: dict[str, dict[str, Any]] = {}
    for session_id, samples in session_samples.items():
        rewards = {float(sample["reward"]["score"]) for sample in samples}
        assert len(rewards) == 1, rewards
        session_rewards[session_id] = rewards.pop()
        session_trace_counts[session_id] = len(samples)
        assert len(samples) == 4, (session_id, len(samples))
        fidelity = validate_session_fidelity(session_id, samples, contexts)
        session_fidelity[session_id] = fidelity
        session_tool_names[session_id] = fidelity["tool_names"]
    assert all(reward == 1.0 for reward in session_rewards.values()), session_rewards
    root_ids = {
        fidelity["root_agent_session_id"] for fidelity in session_fidelity.values()
    }
    child_links = {
        (
            fidelity["child_agent_session_id"],
            fidelity["child_parent_session_id"],
        )
        for fidelity in session_fidelity.values()
    }
    assert len(root_ids) == EXPECTED_SESSIONS, root_ids
    assert len(child_links) == EXPECTED_SESSIONS, child_links
    child_ids = {child for child, _ in child_links}
    assert len(child_ids) == EXPECTED_SESSIONS, child_links
    assert root_ids.isdisjoint(child_ids), (root_ids, child_ids)
    assert {parent for _, parent in child_links} == root_ids, (root_ids, child_links)

    train_hashes = [token_hash(tokens) for tokens in train["tokens"]]
    assert Counter(train_hashes) == Counter(raw_hashes), (train_hashes, raw_hashes)
    train_weight_versions = [
        tuple(str(value) for value in versions) for versions in train["weight_versions"]
    ]
    assert all(versions == ("1",) for versions in train_weight_versions), (
        train_weight_versions
    )
    train_token_versions = list(zip(train_hashes, train_weight_versions, strict=True))
    raw_token_versions = list(zip(raw_hashes, raw_weight_versions, strict=True))
    assert Counter(train_token_versions) == Counter(raw_token_versions), (
        train_token_versions,
        raw_token_versions,
    )
    train_sample_indices = [int(value) for value in train["sample_indices"]]
    train_rollout_ids = [int(value) for value in train["rollout_ids"]]
    assert len(train_sample_indices) == len(raw_samples)
    assert len(train_rollout_ids) == len(raw_samples)
    assert len(set(train_sample_indices)) == EXPECTED_SESSIONS
    assert set(train_rollout_ids) == raw_rollout_ids
    train_identities = list(
        zip(
            train_sample_indices,
            train_rollout_ids,
            train_hashes,
            train_weight_versions,
            strict=True,
        )
    )
    assert Counter(train_identities) == Counter(raw_identities), (
        train_identities,
        raw_identities,
    )
    train_raw_rewards = [float(value) for value in train["raw_reward"]]
    assert all(value == 1.0 for value in train_raw_rewards), train_raw_rewards

    advantages = [tensor.detach().cpu().float() for tensor in train["advantages"]]
    assert advantages
    assert all(torch.isfinite(tensor).all().item() for tensor in advantages)
    assert any(torch.count_nonzero(tensor).item() > 0 for tensor in advantages)
    assert all(float(mask_sum) > 0 for mask_sum in train["rollout_mask_sums"])

    event_path = require_fresh(dump / "events" / "actor_cell0_rank0.jsonl")
    events = [
        json.loads(line) for line in event_path.read_text().splitlines() if line.strip()
    ]
    train_metrics = [
        event["metrics"]
        for event in events
        if isinstance(event.get("metrics"), dict)
        and "train/grad_norm" in event["metrics"]
    ]
    assert len(train_metrics) == 1, train_metrics
    grad_norm = float(train_metrics[-1]["train/grad_norm"])
    assert math.isfinite(grad_norm) and grad_norm > 0, grad_norm

    driver_log_path = require_fresh(Path(cli.driver_log))
    driver_log = driver_log_path.read_text(errors="replace")
    valid_marker = "op=train_step rollout=0 step=0 attempt=0 outcome=NORMAL valid_step=true"
    valid_step_matches = list(
        re.finditer(
            rf"actor_cell0_rank(\d+)\].*{re.escape(valid_marker)}", driver_log
        )
    )
    assert Counter(match.group(1) for match in valid_step_matches) == Counter(
        str(rank) for rank in range(EXPECTED_TRAIN_RANKS)
    ), [match.group(0) for match in valid_step_matches]
    assert len(re.findall(r"op=train_step\s", driver_log)) == EXPECTED_TRAIN_RANKS, (
        len(re.findall(r"op=train_step\s", driver_log)),
        EXPECTED_TRAIN_RANKS,
    )
    assert re.search(r"use_rollout_routing_replay\s+\.+\s+True", driver_log)
    expected_config = {
        "fp4_recipe": "nvfp4",
        "fp4": "e2m1",
        "transformer_impl": "transformer_engine",
        "hf_checkpoint": "/opt/models/Qwen3-30B-A3B-NVFP4/",
        "seq_length": "131072",
        "rollout_max_context_len": "131072",
        "tensor_model_parallel_size": "1",
        "context_parallel_size": "4",
        "pipeline_model_parallel_size": "1",
        "expert_model_parallel_size": "4",
        "expert_tensor_parallel_size": "1",
        "max_tokens_per_gpu": "32768",
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": "1",
        "optimizer_cpu_offload": "True",
        "overlap_cpu_optimizer_d2h_h2d": "True",
        "use_precision_aware_optimizer": "True",
        "position_embedding_type": "yarn",
        "rotary_scaling_factor": "4.0",
        "yarn_original_max_position_embeddings": "32768",
    }
    for name, value in expected_config.items():
        assert re.search(rf"{name}\s+\.+\s+{re.escape(value)}", driver_log), (name, value)
    assert (
        "Applied bridge YaRN config: seq_length=131072 scaling_factor=4.0 "
        "original_max_position_embeddings=32768"
    ) in driver_log
    assert re.search(r"te_precision_config_file\s+\.+\s+base64:", driver_log)
    assert "rollout_routed_experts" not in train
    registered = re.findall(
        r"Replay data registered: data_key=rollout_routed_experts modules=(\d+) "
        r"streams=(\d+) microbatches=(\d+) entries_per_module=(\d+)",
        driver_log,
    )
    assert registered, "missing routing replay registration evidence"
    assert all(int(modules) == 48 and int(streams) == 48 for modules, streams, _, _ in registered), registered
    consumed = re.findall(
        r"Rollout replay consumed: name=routing modules=(\d+) "
        r"min_entries=(\d+) max_entries=(\d+) min_forward=(\d+) "
        r"max_forward=(\d+) min_backward=(\d+) max_backward=(\d+)",
        driver_log,
    )
    assert consumed, "missing routing replay consumption evidence"
    assert all(
        int(modules) == 48
        and int(min_entries) > 0
        and min_entries == max_entries == min_forward == max_forward == min_backward == max_backward
        for (
            modules,
            min_entries,
            max_entries,
            min_forward,
            max_forward,
            min_backward,
            max_backward,
        ) in consumed
    ), consumed
    valid_start = min(match.start() for match in valid_step_matches)
    valid_end = max(match.end() for match in valid_step_matches)
    sync_positions = [
        match.start()
        for match in re.finditer(
            r"actor_cell0_rank0\].*fn=update_weights phase=end ok=true", driver_log
        )
    ]
    assert len(sync_positions) == 2, sync_positions
    assert any(position < valid_start for position in sync_positions), sync_positions
    assert any(position > valid_end for position in sync_positions), sync_positions
    assert "successfully saved checkpoint from iteration" in driver_log
    checkpoint = require_fresh(
        run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    )
    assert checkpoint.read_text().strip() == EXPECTED_CHECKPOINT_ITERATION, (
        checkpoint.read_text()
    )
    pre_server_info_path = require_fresh(Path(cli.pre_server_info))
    pre_server_info = json.loads(pre_server_info_path.read_text())
    assert str(pre_server_info.get("weight_version")) == "0", pre_server_info
    with urllib.request.urlopen(cli.server_info_url, timeout=30) as response:
        post_server_info = json.load(response)
    assert str(post_server_info.get("weight_version")) == "2", post_server_info

    evidence_dir = Path(cli.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "result": "passed",
        "boundary": "four real ProRL SWE-Gym trajectories joined exactly to Miles NVFP4 optimizer input",
        "instance_id": INSTANCE_ID,
        "sample_count": len(raw_samples),
        "session_count": len(raw_session_ids),
        "session_ids": sorted(raw_session_ids),
        "rollout_ids": sorted(raw_rollout_ids),
        "session_trace_counts": session_trace_counts,
        "session_rewards": session_rewards,
        "reward_counts": dict(Counter(session_rewards.values())),
        "session_tool_names": session_tool_names,
        "session_fidelity": session_fidelity,
        "agent_contexts": {
            f"{session_id}:{trace_index}": context
            for (session_id, trace_index), context in sorted(contexts.items())
        },
        "dynamo_request_matches": matched_trace_records,
        "fresh_dynamo_trace_record_count": len(trace_records),
        "trace_start_line_counts": trace_start,
        "session_trajectory_identities": {
            session_id: list(identity)
            for session_id, identity in sorted(session_identity.items())
        },
        "evaluations": evaluations,
        "raw_samples": raw_summaries,
        "train_token_sha256": train_hashes,
        "raw_weight_versions": [list(values) for values in raw_weight_versions],
        "train_weight_versions": [list(values) for values in train_weight_versions],
        "token_weight_version_pairs": [
            [digest, list(versions)] for digest, versions in train_token_versions
        ],
        "raw_trajectory_identities": [
            [sample_index, rollout_id, digest, list(versions)]
            for sample_index, rollout_id, digest, versions in raw_identities
        ],
        "train_trajectory_identities": [
            [sample_index, rollout_id, digest, list(versions)]
            for sample_index, rollout_id, digest, versions in train_identities
        ],
        "sample_indices": train_sample_indices,
        "train_rollout_ids": train_rollout_ids,
        "raw_reward": train_raw_rewards,
        "advantages_nonzero": [
            int(torch.count_nonzero(tensor).item()) for tensor in advantages
        ],
        "gradient_norm": grad_norm,
        "valid_step": True,
        "routing_replay_consumed": True,
        "routing_replay_registration": registered,
        "routing_replay_consumption": consumed,
        "nvfp4_training_config": expected_config,
        "pre_step_server_info": pre_server_info,
        "post_step_sync": True,
        "post_step_server_info": post_server_info,
        "attempt_start_ns": attempt_start_ns,
        "checkpoint_iteration": checkpoint.read_text().strip(),
    }
    (evidence_dir / "swegym-train-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

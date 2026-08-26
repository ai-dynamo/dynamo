#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public ProRL-to-Miles rollout function with a real Agent subagent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.types import Sample
from slime_bridge.rollout import generate_rollout_polar_async, stop_global_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-config", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--rollout-id", type=int, default=9000)
    parser.add_argument("--expected-weight-version", default="0")
    return parser.parse_args()


class ReplenishingSource:
    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.calls = 0

    def get_samples(self, count: int) -> list[list[Sample]]:
        assert count == 1, count
        index = self.calls
        self.calls += 1
        return [[
            Sample(
                prompt=self.instruction,
                label="",
                metadata={"integration_case": "m5-bridge-smoke"},
                group_index=0,
                index=index,
            )
        ]]


def token_hash(tokens: list[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode()
    return hashlib.sha256(payload).hexdigest()


def find_agent_contexts(value: Any) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    if isinstance(value, dict):
        candidate = value.get("agent_context")
        if isinstance(candidate, dict) and candidate.get("session_id"):
            contexts.append(
                {str(k): str(v) for k, v in candidate.items() if v is not None}
            )
        for child in value.values():
            contexts.extend(find_agent_contexts(child))
    elif isinstance(value, list):
        for child in value:
            contexts.extend(find_agent_contexts(child))
    return contexts


def snapshot_trace_line_counts(trace_dir: Path) -> dict[str, int]:
    return {
        str(path): len(path.read_text(errors="replace").splitlines())
        for path in sorted(trace_dir.rglob("*.jsonl"))
    }


def load_trace_contexts(
    trace_dir: Path,
    start_line_counts: dict[str, int],
    minimum_contexts: int,
    timeout: float = 45.0,
) -> tuple[list[dict[str, str]], list[dict]]:
    deadline = time.monotonic() + timeout
    last_records: list[dict] = []
    last_contexts: list[dict[str, str]] = []
    while time.monotonic() < deadline:
        records: list[dict] = []
        for path in sorted(trace_dir.rglob("*.jsonl")):
            lines = path.read_text(errors="replace").splitlines()
            start_line = start_line_counts.get(str(path), 0)
            for line in lines[start_line:]:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        contexts = [
            context for record in records for context in find_agent_contexts(record)
        ]
        triggers = {context.get("input_trigger") for context in contexts}
        if len(contexts) >= minimum_contexts and "tool_result" in triggers:
            return contexts, records
        last_records, last_contexts = records, contexts
        time.sleep(1.0)
    return last_contexts, last_records


def main() -> None:
    cli = parse_args()
    config = yaml.safe_load(Path(cli.polar_config).read_text())
    values = {
        **config,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "update_weights_interval": 1,
        "max_tokens_per_gpu": 32768,
        "context_parallel_size": 1,
        "num_layers": 48,
        "use_rollout_routing_replay": True,
        "start_rollout_id": cli.rollout_id,
        "hf_checkpoint": "Qwen/Qwen3-30B-A3B",
    }
    args = SimpleNamespace(**values)
    source = ReplenishingSource(cli.instruction)
    trace_dir = Path(cli.trace_dir)
    trace_start_line_counts = snapshot_trace_line_counts(trace_dir)
    try:
        output = generate_rollout_polar_async(
            args, cli.rollout_id, source, evaluation=False
        )
    finally:
        stop_global_worker()

    assert isinstance(output, RolloutFnTrainOutput), type(output)
    assert len(output.samples) == 1, len(output.samples)
    samples = output.samples[0]
    assert len(samples) == 3, (
        "Agent execution must produce root, child, and resumed-root traces"
    )

    session_ids: set[str] = set()
    rollout_ids: set[int] = set()
    summaries: list[dict[str, Any]] = []
    samples_by_trace: dict[int, Sample] = {}
    for sample in samples:
        assert isinstance(sample, Sample), type(sample)
        sample.validate()
        assert sample.status is Sample.Status.COMPLETED, sample.status
        assert not sample.remove_sample
        assert sample.response_length > 0
        assert len(sample.tokens) > sample.response_length
        assert len(sample.loss_mask or []) == sample.response_length
        assert sum(sample.loss_mask or []) > 0
        assert len(sample.rollout_log_probs or []) == sample.response_length
        assert all(
            math.isfinite(float(value)) for value in sample.rollout_log_probs or []
        )
        assert float(sample.reward["score"]) == 1.0
        assert sample.weight_versions == [cli.expected_weight_version], (
            sample.weight_versions
        )

        routed = np.asarray(sample.rollout_routed_experts)
        assert routed.shape == (len(sample.tokens) - 1, 48, 8), routed.shape
        assert routed.dtype == np.int32, routed.dtype
        assert int(routed.min()) >= 0
        assert int(routed.max()) < 128

        polar = sample.metadata["polar"]
        trace_index = int(polar["trace_index"])
        samples_by_trace[trace_index] = sample
        session_ids.add(str(polar["session_id"]))
        rollout_ids.add(int(sample.rollout_id))
        summaries.append(
            {
                "trace_index": trace_index,
                "session_id": str(polar["session_id"]),
                "task_id": str(polar["task_id"]),
                "status": sample.status.value,
                "token_count": len(sample.tokens),
                "response_length": sample.response_length,
                "active_loss_tokens": int(sum(sample.loss_mask or [])),
                "token_sha256": token_hash(sample.tokens),
                "weight_versions": sample.weight_versions,
                "routed_experts_shape": list(routed.shape),
                "routed_experts_min": int(routed.min()),
                "routed_experts_max": int(routed.max()),
                "response": sample.response,
                "finish_reason": polar["trace_debug"]["finish_reason"],
            }
        )

    assert len(session_ids) == 1, session_ids
    assert len(rollout_ids) == 1, rollout_ids
    assert sorted(samples_by_trace) == [0, 1, 2], sorted(samples_by_trace)
    root = samples_by_trace[0]
    child = samples_by_trace[1]
    resumed_root = samples_by_trace[2]
    root_debug = root.metadata["polar"]["trace_debug"]
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
    child_prompt = json.dumps(child.prompt, ensure_ascii=False)
    assert "You are an agent for Claude Code" in child_prompt, child_prompt
    assert "Calculate 13 * 17 and return exactly CHILD_221." in child_prompt
    assert "CHILD_221" in child.response, child.response
    resumed_prompt = json.dumps(resumed_root.prompt, ensure_ascii=False)
    assert json.dumps({"role": "tool"})[1:-1] in resumed_prompt, resumed_prompt
    assert agent_call["id"] in resumed_prompt, resumed_prompt
    assert "CHILD_221" in resumed_prompt and "agentId:" in resumed_prompt
    assert "ROOT_OK CHILD_221" in resumed_root.response, resumed_root.response

    contexts, trace_records = load_trace_contexts(
        trace_dir, trace_start_line_counts, minimum_contexts=len(samples)
    )
    assert len(contexts) == 3, contexts
    assert [context.get("input_trigger") for context in contexts] == [
        "user_message",
        "user_message",
        "tool_result",
    ], contexts
    root_context, child_context, resumed_root_context = contexts
    assert root_context.get("parent_session_id") is None, root_context
    assert child_context.get("parent_session_id") == root_context["session_id"], (
        contexts
    )
    assert child_context["session_id"] != root_context["session_id"], contexts
    assert resumed_root_context.get("parent_session_id") is None, resumed_root_context
    assert resumed_root_context["session_id"] == root_context["session_id"], contexts
    lineage_mode = "explicit_parent_session_id"

    evidence_dir = Path(cli.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "result": "passed",
        "boundary": (
            "Miles public rollout function -> ProRL real Claude Code Agent -> "
            "Dynamo chat completions"
        ),
        "sample_count": len(samples),
        "session_ids": sorted(session_ids),
        "rollout_ids": sorted(rollout_ids),
        "metrics": output.metrics,
        "source_calls": source.calls,
        "samples": summaries,
        "agent_contexts": contexts,
        "lineage_mode": lineage_mode,
        "dynamo_trace_record_count": len(trace_records),
        "trace_start_line_counts": trace_start_line_counts,
    }
    (evidence_dir / "bridge-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (evidence_dir / "agent-contexts.json").write_text(
        json.dumps(contexts, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

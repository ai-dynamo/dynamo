#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove four real ProRL SWE-Gym trajectories traverse Dynamo correctly."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from enforce_root_final import POST_AGENT_CONTEXT
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.types import Sample
from prepare_swegym_task import (
    AGENT_INSTRUCTION,
    BASH_COMMAND,
    CHILD_SUCCESS_HANDOFF,
    ROOT_FINAL_INSTRUCTION,
    ROOT_SUCCESS_FINAL,
)
from slime_bridge.rollout import generate_rollout_polar_async, stop_global_worker
from swegym_fidelity import (
    load_and_match_trace_records,
    snapshot_trace_line_counts,
    tool_names,
    validate_session_fidelity,
)

INSTANCE_ID = "getmoto__moto-7365"
EXPECTED_SESSIONS = 4
MAX_SEQUENCE_LENGTH = 131072
MAX_RESPONSE_LENGTH = 4096
EXPECTED_PATCH_SHA256 = "bd25ef8a5b93198b2eccb99d9ed8bcbc2275ebfba50695df386a26f40fe8b6ce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-config", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--rollout-id", type=int, default=9000)
    parser.add_argument("--expected-weight-version", default="0")
    parser.add_argument("--expected-sessions", type=int, choices=(1, 4), default=4)
    return parser.parse_args()


def load_task(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1, len(rows)
    row = rows[0]
    assert row["metadata"]["instance_id"] == INSTANCE_ID, row["metadata"]
    prompt_content = row["prompt"][0]["content"]
    assert prompt_content.startswith(AGENT_INSTRUCTION)
    assert prompt_content.endswith(ROOT_FINAL_INSTRUCTION)
    instance = row["metadata"]["instance"]
    assert instance["FAIL_TO_PASS"] == [
        "tests/test_dynamodb/test_dynamodb_update_expressions.py::test_update_item_add_float"
    ]
    assert instance["PASS_TO_PASS"] == [
        "tests/test_dynamodb/test_dynamodb_update_expressions.py::test_update_different_map_elements_in_single_request"
    ]
    return row


class SingleTaskSource:
    def __init__(self, row: dict[str, Any], expected_sessions: int) -> None:
        self.row = row
        self.expected_sessions = expected_sessions
        self.calls = 0

    def get_samples(self, count: int) -> list[list[Sample]]:
        assert count == 1, count
        assert self.calls == 0, self.calls
        self.calls += 1
        return [[
            Sample(
                prompt=self.row["prompt"],
                label=self.row.get("label", ""),
                metadata=self.row["metadata"],
                group_index=0,
                index=index,
            )
            for index in range(self.expected_sessions)
        ]]


def token_hash(tokens: list[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluation_for(sample: Sample) -> dict[str, Any]:
    polar = sample.metadata["polar"]
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
    reward = float(sample.reward["score"])
    assert reward in (0.0, 1.0), reward
    assert report["resolved"] is True, report
    assert reward == 1.0, reward
    return evaluation


def validate_config_contract(config: dict[str, Any]) -> None:
    task_template = config["polar_task_template"]
    assert task_template["builder"]["strategy"] == "per_request"
    assert task_template["evaluator"]["refresh_runtime"] is True
    expected_baseline = "7f6c9cb1deafb280fe7fcc7551c38e397f11a706"
    prepare_command = task_template["runtime"]["prepare"][0]["command"]
    assert f'test "$(git rev-parse HEAD)" = {expected_baseline}' in prepare_command
    for required in (
        "grep -qxF '/.claude/' .git/info/exclude",
        "m6-general-purpose-agent.md .claude/agents/general-purpose.md",
        "m6-claude-settings.json .claude/settings.json",
        "guard_root_bash.py --self-test",
        "enforce_root_final.py --self-test",
        "'\"permissionDecision\":\"deny\"'",
        "'\"permissionDecision\":\"allow\"'",
        "'\"additionalContext\":\"M6_POST_AGENT_FINALIZATION.",
        'agent_id="child-preflight"',
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1 claude agents",
        "grep -F '1 active agents'",
        "grep -F 'general-purpose'",
        "! grep -F 'Explore'",
        'test -z "$(git status --porcelain)"',
    ):
        assert required in prepare_command, required
    assert not re.findall(r"{([^{}]+)}", prepare_command), prepare_command
    for required in (
        'json.dumps(dict(hook_event_name="PreToolUse", tool_name="Bash"',
        'json.dumps(dict(hook_event_name="PostToolUse", tool_name="Agent"',
        'agent_id="child-preflight"',
        'tool_input=dict(command="git status")',
    ):
        assert required in prepare_command, required
    runtime_volumes = task_template["runtime"]["kwargs"]["volumes"]
    assert (
        "/shared/test-artifacts/prorl-miles-nvfp4-m6-f049b16-20260826t052909z/"
        "integration:/opt/m6-integration:ro"
    ) in runtime_volumes
    assert task_template["evaluator"]["config"]["patch_command"] == (
        "cd /polar/session/workspace && git add -A && git diff --cached --binary "
        f"{expected_baseline} --"
    )
    runtime_env = task_template["runtime"]["env"]
    agent_env = task_template["agent"]["env"]
    agent_settings = task_template["agent"]["settings"]
    assert runtime_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"
    assert agent_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "122880"
    assert agent_env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "95"
    assert agent_env["CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY"] == "1"
    assert agent_env["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"] == "1"
    assert agent_settings["max_turns"] == 2
    assert agent_settings["max_thinking_tokens"] == 1024
    assert agent_settings["tools"] == "Agent,Bash"
    assert agent_settings["allowed_tools"] == "Agent"
    system_prompt = agent_settings["append_system_prompt"]
    for required in (
        "Make exactly one root tool call: one foreground Agent subagent",
        "subagent_type=general-purpose",
        "Agent is the root's only permitted tool",
        "Bash is present in the shared registry so the child can receive it",
        "PreToolUse hook blocks root Bash",
        "Do not set isolation or create a worktree",
        "operate on the current checkout",
        "direct policy permits only Bash",
        "/opt/m6-integration/apply_moto_fix.sh",
        "M6_CHILD_DONE_VALIDATED",
        "The child owns all compilation, test, diff, and git-status verification",
        "exactly five sites",
        "cast_value fallback near line 149",
        "do not modify Item ADD set branches",
        "fresh grader supplies the hidden target",
        "do not reason about or verify the task",
        "do not call Agent, Bash, or any tool",
        "Output exactly: M6 repair complete.",
        "Do not commit, push, fetch, or modify Git remotes",
    ):
        assert required in system_prompt, required
    assert BASH_COMMAND in system_prompt, (BASH_COMMAND, system_prompt)
    agent_definition = Path(__file__).with_name("m6-general-purpose-agent.md").read_text()
    for required in (
        "name: general-purpose",
        "description: Run exactly /opt/m6-integration/apply_moto_fix.sh",
        "tools: Bash",
        "model: inherit",
        "maxTurns: 2",
        BASH_COMMAND,
        "Make exactly one real",
        "__M6_BASH_VALIDATION_PASS__",
        "Do not invoke any other command or tool",
        CHILD_SUCCESS_HANDOFF,
        "Do not include a command, path, suggestion, or any other text",
    ):
        assert required in agent_definition, required
    assert ROOT_SUCCESS_FINAL in system_prompt
    assert POST_AGENT_CONTEXT.startswith("M6_POST_AGENT_FINALIZATION.")
    hook_settings = json.loads(
        Path(__file__).with_name("m6-claude-settings.json").read_text()
    )
    assert hook_settings == {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python3 /opt/m6-integration/guard_root_bash.py",
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Agent",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python3 /opt/m6-integration/enforce_root_final.py",
                            "timeout": 10,
                        }
                    ],
                }
            ],
        }
    }, hook_settings
    guard = Path(__file__).with_name("guard_root_bash.py").read_text()
    for required in (
        'EXPECTED_COMMAND = "/opt/m6-integration/apply_moto_fix.sh"',
        'elif not agent_id:',
        'elif command != EXPECTED_COMMAND:',
        '"hookEventName": "PreToolUse"',
        '"permissionDecision": decision',
        '__M6_ROOT_BASH_GUARD_SELF_TEST_PASS__',
    ):
        assert required in guard, required
    finalizer = Path(__file__).with_name("enforce_root_final.py").read_text()
    for required in (
        'event_name != "PostToolUse"',
        'tool_name != "Agent"',
        "elif agent_id:",
        "CHILD_SUCCESS_HANDOFF not in response_blob",
        '"hookEventName": "PostToolUse"',
        '"additionalContext": POST_AGENT_CONTEXT',
        '"continue": False',
        "__M6_ROOT_FINALIZER_SELF_TEST_PASS__",
    ):
        assert required in finalizer, required
    helper = Path(__file__).with_name("apply_moto_fix.sh").read_text()
    for required in (
        "set -euo pipefail",
        'source.count("float(self.value)") == 3',
        'source.count("float(other.value)") == 2',
        "python -m py_compile moto/dynamodb/models/dynamo_type.py",
        "pytest -q -k test_update_item",
        "git diff --check",
        'test "$(git diff --name-only)" = moto/dynamodb/models/dynamo_type.py',
        "__M6_BASH_VALIDATION_PASS__",
    ):
        assert required in helper, required
    assert int(agent_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) + int(
        runtime_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]
    ) == 131072


def main() -> None:
    cli = parse_args()
    row = load_task(Path(cli.prompt_data))
    config = yaml.safe_load(Path(cli.polar_config).read_text())
    validate_config_contract(config)
    args = SimpleNamespace(
        **config,
        rollout_batch_size=1,
        n_samples_per_prompt=cli.expected_sessions,
        update_weights_interval=1,
        max_tokens_per_gpu=131072,
        context_parallel_size=1,
        num_layers=48,
        use_rollout_routing_replay=True,
        start_rollout_id=cli.rollout_id,
        hf_checkpoint="Qwen/Qwen3-30B-A3B",
    )
    source = SingleTaskSource(row, cli.expected_sessions)
    trace_dir = Path(cli.trace_dir)
    trace_start = snapshot_trace_line_counts(trace_dir)
    try:
        output = generate_rollout_polar_async(args, cli.rollout_id, source, evaluation=False)
    finally:
        stop_global_worker()

    assert isinstance(output, RolloutFnTrainOutput), type(output)
    assert len(output.samples) == 1, len(output.samples)
    samples = output.samples[0]
    assert len(samples) == cli.expected_sessions * 4, len(samples)

    session_samples: dict[str, list[Sample]] = defaultdict(list)
    session_identities: dict[str, set[tuple[int, int]]] = defaultdict(set)
    summaries: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for sample in samples:
        assert isinstance(sample, Sample), type(sample)
        sample.validate()
        assert len(sample.tokens) <= MAX_SEQUENCE_LENGTH, len(sample.tokens)
        assert sample.status is Sample.Status.COMPLETED, sample.status
        assert not sample.remove_sample
        assert sample.response_length > 0
        assert sample.response_length <= MAX_RESPONSE_LENGTH, sample.response_length
        assert len(sample.loss_mask or []) == sample.response_length
        assert sum(sample.loss_mask or []) > 0
        assert len(sample.rollout_log_probs or []) == sample.response_length
        assert all(math.isfinite(float(value)) for value in sample.rollout_log_probs or [])
        assert sample.weight_versions == [cli.expected_weight_version], sample.weight_versions
        routed = np.asarray(sample.rollout_routed_experts)
        assert routed.shape == (len(sample.tokens) - 1, 48, 8), routed.shape
        assert routed.dtype == np.int32
        assert int(routed.min()) >= 0 and int(routed.max()) < 128

        polar = sample.metadata["polar"]
        assert polar["trajectory_metadata"]["builder"] == "per_request"
        assert polar["result_metadata"]["instance_id"] == INSTANCE_ID
        session_id = str(polar["session_id"])
        session_samples[session_id].append(sample)
        session_identities[session_id].add((int(sample.index), int(sample.rollout_id)))
        evaluations[session_id] = evaluation_for(sample)
        summaries.append(
            {
                "session_id": session_id,
                "rollout_id": int(sample.rollout_id),
                "trace_index": int(polar["trace_index"]),
                "finish_reason": polar["trace_debug"]["finish_reason"],
                "tool_names": tool_names(sample),
                "reward": float(sample.reward["score"]),
                "token_count": len(sample.tokens),
                "response_length": sample.response_length,
                "active_loss_tokens": int(sum(sample.loss_mask or [])),
                "token_sha256": token_hash(sample.tokens),
                "weight_versions": sample.weight_versions,
                "routed_experts_shape": list(routed.shape),
            }
        )

    assert len(session_samples) == cli.expected_sessions, session_samples.keys()
    assert len({int(sample.rollout_id) for sample in samples}) == cli.expected_sessions
    assert all(len(identities) == 1 for identities in session_identities.values()), (
        session_identities
    )
    session_identity = {
        session_id: next(iter(identities))
        for session_id, identities in session_identities.items()
    }
    assert len(set(session_identity.values())) == cli.expected_sessions, session_identity
    contexts, matched_trace_records, trace_records = load_and_match_trace_records(
        trace_dir, trace_start, samples
    )
    session_rewards: dict[str, float] = {}
    session_tool_names: dict[str, list[str]] = {}
    session_trace_counts: dict[str, int] = {}
    session_fidelity: dict[str, dict[str, Any]] = {}
    for session_id, traces in session_samples.items():
        rewards = {float(trace.reward["score"]) for trace in traces}
        assert len(rewards) == 1, rewards
        session_rewards[session_id] = rewards.pop()
        session_trace_counts[session_id] = len(traces)
        assert len(traces) == 4, (session_id, len(traces))
        fidelity = validate_session_fidelity(session_id, traces, contexts)
        session_fidelity[session_id] = fidelity
        session_tool_names[session_id] = fidelity["tool_names"]
    assert sum(reward == 1.0 for reward in session_rewards.values()) == cli.expected_sessions

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
    assert len(root_ids) == cli.expected_sessions, root_ids
    assert len(child_links) == cli.expected_sessions, child_links
    child_ids = {child for child, _ in child_links}
    assert len(child_ids) == cli.expected_sessions, child_links
    assert root_ids.isdisjoint(child_ids), (root_ids, child_ids)
    assert {parent for _, parent in child_links} == root_ids, (root_ids, child_links)
    linked_root_parents = {parent for _, parent in child_links}
    assert len(linked_root_parents) == cli.expected_sessions, linked_root_parents

    evidence_dir = Path(cli.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "result": "passed",
        "boundary": f"{cli.expected_sessions} ProRL SWE-Gym coding session(s) -> Dynamo chat traces -> real fresh-runtime grading",
        "instance_id": INSTANCE_ID,
        "sample_count": len(samples),
        "session_count": len(session_samples),
        "session_trace_counts": session_trace_counts,
        "session_trajectory_identities": {
            session_id: list(identity)
            for session_id, identity in sorted(session_identity.items())
        },
        "session_rewards": session_rewards,
        "reward_counts": dict(Counter(session_rewards.values())),
        "session_tool_names": session_tool_names,
        "session_fidelity": session_fidelity,
        "evaluations": evaluations,
        "samples": summaries,
        "metrics": output.metrics,
        "source_calls": source.calls,
        "agent_contexts": {
            f"{session_id}:{trace_index}": context
            for (session_id, trace_index), context in sorted(contexts.items())
        },
        "dynamo_request_matches": matched_trace_records,
        "root_agent_session_count": len(root_ids),
        "distinct_linked_root_parent_count": len(linked_root_parents),
        "linked_root_parent_ids": sorted(linked_root_parents),
        "child_agent_links": sorted([list(link) for link in child_links]),
        "dynamo_trace_record_count": len(trace_records),
        "trace_start_line_counts": trace_start,
    }
    (evidence_dir / "swegym-rollout-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

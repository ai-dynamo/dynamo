# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .cli import DEFAULT_PROFILES, _bfcl_counts, _profile, _run_custom, main


def test_qualification_profile_has_bounded_stratified_coverage() -> None:
    profile, profile_hash = _profile(DEFAULT_PROFILES, "qualification")

    assert sum(profile["bfcl"]["categories"].values()) == 200
    assert set(profile["bfcl"]["categories"]) >= {
        "simple_python",
        "parallel_multiple",
        "irrelevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
    }
    assert sum(len(values) for values in profile["tau2"]["tasks"].values()) == 9
    assert set(profile["tau2"]["tasks"]) == {"airline", "retail", "telecom"}
    assert profile["custom"]["case_profile"] == "auto"
    assert profile["custom"]["modes"] == ["nonstream", "stream"]
    assert profile["custom"]["concurrency"] == 8
    assert len(profile_hash) == 64


def test_all_public_profiles_resolve_to_the_same_qualification_budget() -> None:
    qualification, _ = _profile(DEFAULT_PROFILES, "qualification")

    for name in ("manual", "pr", "nightly"):
        selected, _ = _profile(DEFAULT_PROFILES, name)
        assert selected == qualification


def test_bfcl_dry_run_writes_normalized_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "bfcl"

    exit_code = main(
        [
            "--suite",
            "bfcl",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "example/model",
            "--runtime",
            "dynamo-vllm",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    assert result["contract_version"] == 1
    assert result["suite"] == "bfcl"
    assert result["execution_status"] == "planned"
    commands = result["artifacts"]["commands"]
    assert len(commands) == 14
    assert all(
        command[command.index("--env") + 1] == "OPENAI_API_KEY" for command in commands
    )
    assert "--partial-eval" in commands[-1]
    assert "@sha256:" in result["provenance"]["runner_image"]


def test_custom_dry_run_uses_dynamo_owned_full_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom"

    exit_code = main(
        [
            "--suite",
            "custom",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "google/gemma-4-31B-it",
            "--runtime",
            "dynamo-vllm",
            "--request-contract-json",
            '{"enabled":{"thinking":true},"disabled":{"thinking":false}}',
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    command = result["artifacts"]["command"]
    assert result["execution_status"] == "planned"
    assert command[1].endswith("benchmarks/tool_calling/custom_runner.py")
    assert "--case-profile" in command
    assert command[command.index("--modes") + 1] == "nonstream,stream"


def test_custom_normalizes_the_complete_detailed_report(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "custom"
    report_path = (
        output_dir / "site" / "models" / "qualification" / "artifacts" / "latest.json"
    )

    def fake_run_command(*_args, **_kwargs):
        report_path.parent.mkdir(parents=True)
        report_path.write_text(
            json.dumps(
                {
                    "config": {
                        "case_profile": "gemma4",
                        "case_ids": ["one", "two"],
                        "modes": ["nonstream", "stream"],
                        "iterations": 1,
                    },
                    "summary": {"passed": 3, "failed": 1, "total": 4},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 1, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        model="google/gemma-4-31B-it",
        runtime="dynamo-vllm",
        output_dir=str(output_dir),
        request_contract_json="{}",
        request_timeout=300.0,
    )
    config, _ = _profile(DEFAULT_PROFILES, "qualification")
    result = {"coverage": dict(config["custom"])}

    _run_custom(args, config["custom"], result)

    assert result["execution_status"] == "complete"
    assert result["verdict"] == "fail"
    assert result["summary"] == {
        "passed": 3,
        "failed": 1,
        "total": 4,
        "completed": 4,
        "score": 0.75,
    }
    assert result["coverage"]["resolved_case_profile"] == "gemma4"


def test_tau2_dry_run_uses_fixed_tasks_and_one_trial(tmp_path: Path) -> None:
    output_dir = tmp_path / "tau2"

    exit_code = main(
        [
            "--suite",
            "tau2",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "example/model",
            "--runtime",
            "dynamo-sglang",
            "--output-dir",
            str(output_dir),
            "--simulator-model",
            "simulator/model",
            "--simulator-base-url",
            "https://simulator.example/v1",
            "--simulator-api-key",
            "do-not-persist-this-secret",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    assert result["execution_status"] == "planned"
    assert len(result["artifacts"]["commands"]) == 3
    assert result["coverage"]["trials"] == 1
    assert "do-not-persist-this-secret" not in json.dumps(result)


def test_bfcl_counts_aggregate_partial_category_score_files(tmp_path: Path) -> None:
    first = tmp_path / "score" / "model" / "BFCL_v3_simple_python_score.json"
    second = tmp_path / "score" / "model" / "BFCL_v3_parallel_score.json"
    first.parent.mkdir(parents=True)
    first.write_text(
        '{"accuracy": 0.8, "correct_count": 24, "total_count": 30}\n{"valid": false}\n',
        encoding="utf-8",
    )
    second.write_text(
        '{"accuracy": 0.5, "correct_count": 10, "total_count": 20}\n',
        encoding="utf-8",
    )

    assert _bfcl_counts(tmp_path) == (34, 50)


def test_tau_rewards_reads_the_normalized_reward_info(tmp_path: Path) -> None:
    result_path = tmp_path / "airline" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "simulations": [
                    {"reward_info": {"reward": 1}},
                    {"reward_info": {"reward": 0}},
                ]
            }
        ),
        encoding="utf-8",
    )

    from .cli import _tau_rewards

    assert _tau_rewards(tmp_path) == [1.0, 0.0]

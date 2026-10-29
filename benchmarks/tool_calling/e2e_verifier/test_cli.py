# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from .cli import (
    DEFAULT_PROFILES,
    _bfcl_cases,
    _bfcl_counts,
    _custom_selection,
    _custom_verdict,
    _openrouter_config,
    _profile,
    _run_bfcl,
    _run_custom,
    _run_openrouter,
    main,
)


def test_qualification_profile_has_bounded_stratified_coverage() -> None:
    profile, profile_hash = _profile(DEFAULT_PROFILES, "qualification")

    bfcl_cases = _bfcl_cases(profile["bfcl"])
    assert sum(len(case_ids) for case_ids in bfcl_cases.values()) == 50
    assert set(bfcl_cases) == {
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
    }
    assert sum(len(values) for values in profile["tau2"]["tasks"].values()) == 9
    assert set(profile["tau2"]["tasks"]) == {"airline", "retail", "telecom"}
    assert profile["custom"]["case_profile"] == "auto"
    custom_selection, selection_hash = _custom_selection(
        profile["custom"], "google/gemma-4-31B-it"
    )
    assert len(custom_selection["case_ids"]) == 61
    assert len(custom_selection["case_groups"]["generic"]) == 61
    assert custom_selection["case_groups"]["model_specific"] == []
    assert (
        sum(case_id.startswith("customer_") for case_id in custom_selection["case_ids"])
        == 8
    )
    assert "customer_truncated_tool_markup_hidden" in custom_selection["case_ids"]
    assert custom_selection["record_count"] == 122
    assert "exclude_cases" not in profile["custom"]
    assert profile["custom"]["modes"] == ["nonstream", "stream"]
    assert profile["custom"]["concurrency"] == 8
    assert len(selection_hash) == 64
    assert len(profile_hash) == 64


def test_all_public_profiles_resolve_to_the_same_qualification_budget() -> None:
    qualification, _ = _profile(DEFAULT_PROFILES, "qualification")

    for name in ("manual", "pr", "nightly"):
        selected, _ = _profile(DEFAULT_PROFILES, name)
        assert selected == qualification


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generic_cases", ["only-one"], "exactly 61 generic case IDs"),
        ("modes", ["nonstream"], "modes must be nonstream,stream"),
        ("iterations", 2, "exactly 1 iteration"),
    ],
)
def test_custom_qualification_budget_is_a_runtime_contract(
    field: str, value, message: str
) -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = {**profile["custom"], field: value}

    with pytest.raises(ValueError, match=message):
        _custom_selection(config, "google/gemma-4-31B-it")


def test_custom_selection_appends_matching_model_specific_cases() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")

    selection, selection_hash = _custom_selection(
        profile["custom"], "moonshotai/Kimi-K2.6"
    )

    assert selection["resolved_case_profile"] == "kimi_k2"
    assert len(selection["case_groups"]["generic"]) == 61
    assert selection["case_groups"]["model_specific"] == [
        "customer_kimi_consume_prior_tool_result",
        "customer_kimi_parallel_weather_final_answer",
    ]
    assert selection["generic_record_count"] == 122
    assert selection["model_specific_record_count"] == 4
    assert selection["record_count"] == 126
    assert len(selection_hash) == 64


def test_muse_uses_complete_profile_without_generic_thinking_controls() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")

    selection, selection_hash = _custom_selection(
        profile["custom"], "meta-models/Muse-Glimmer-30B"
    )

    assert selection["resolved_case_profile"] == "muse_glimmer"
    assert selection["case_groups"]["generic"] == []
    assert selection["case_groups"]["model_specific"] == []
    assert len(selection["case_groups"]["complete_profile"]) == 61
    assert len(selection["case_ids"]) == 61
    assert selection["generic_case_count"] == 0
    assert selection["model_specific_case_count"] == 0
    assert selection["complete_profile_case_count"] == 61
    assert selection["generic_record_count"] == 0
    assert selection["model_specific_record_count"] == 0
    assert selection["complete_profile_record_count"] == 122
    assert selection["record_count"] == 122
    assert selection["temperature"] == 1.0
    assert len(selection_hash) == 64


def test_qwen36_27b_uses_complete_qwen_xml_profile() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")

    selection, selection_hash = _custom_selection(profile["custom"], "Qwen/Qwen3.6-27B")

    assert selection["resolved_case_profile"] == "qwen3_coder_xml"
    assert selection["case_groups"]["generic"] == []
    assert selection["case_groups"]["model_specific"] == []
    assert len(selection["case_groups"]["complete_profile"]) == 63
    assert len(selection["case_ids"]) == 63
    assert selection["complete_profile_case_count"] == 63
    assert selection["complete_profile_record_count"] == 126
    assert selection["record_count"] == 126
    assert selection["temperature"] == 0.0
    assert len(selection_hash) == 64


def test_minimax_m3_resolves_to_its_own_profile_with_generic_cases() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")

    selection, selection_hash = _custom_selection(
        profile["custom"], "nvidia/MiniMax-M3-NVFP4"
    )

    assert selection["resolved_case_profile"] == "minimax_m3"
    assert len(selection["case_groups"]["generic"]) == 61
    assert selection["case_groups"]["model_specific"] == []
    assert selection["case_groups"]["complete_profile"] == []
    assert selection["record_count"] == 122
    assert len(selection_hash) == 64


def test_custom_selection_rejects_cross_group_overlap() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = {
        **profile["custom"],
        "model_specific_cases": {
            **profile["custom"]["model_specific_cases"],
            "gemma4": [profile["custom"]["generic_cases"][0]],
        },
    }

    with pytest.raises(ValueError, match="both generic and gemma4 model-specific"):
        _custom_selection(config, "google/gemma-4-31B-it")


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
    assert len(commands) == 3
    assert all(
        command[command.index("--env") + 1] == "OPENAI_API_KEY"
        for command in commands[1:]
    )
    assert commands[0][-3:-1] == ["python", "-c"]
    assert "--run-ids" in commands[1]
    assert "--limit" not in commands[1]
    assert "/v1/chat/completions" in commands[1][commands[1].index("--model-args") + 1]
    assert "--partial-eval" not in commands[-1]
    selection = json.loads((output_dir / "bfcl-case-ids.json").read_text())
    assert sum(len(case_ids) for case_ids in selection.values()) == 50
    assert result["coverage"]["resolved_case_count"] == 50
    assert len(result["provenance"]["selection_hash"]) == 64
    assert "@sha256:" in result["provenance"]["runner_image"]


@pytest.mark.parametrize("runtime", ("vllm-serve", "sglang-serve"))
def test_native_runtime_labels_round_trip_in_suite_result(
    tmp_path: Path, runtime: str
) -> None:
    output_dir = tmp_path / runtime

    exit_code = main(
        [
            "--suite",
            "bfcl",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "example/model",
            "--runtime",
            runtime,
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    assert result["runtime"] == runtime
    assert result["execution_status"] == "planned"


def test_bfcl_completes_only_after_exact_generation_and_scoring(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "bfcl"
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = profile["bfcl"]
    cases = _bfcl_cases(config)

    def fake_run_command(command, **_kwargs):
        if "generate" in command:
            result_root = output_dir / "result" / "example_model"
            result_root.mkdir(parents=True)
            for category, case_ids in cases.items():
                path = result_root / f"BFCL_v3_{category}_result.json"
                path.write_text(
                    "\n".join(
                        json.dumps({"id": case_id, "result": []})
                        for case_id in case_ids
                    ),
                    encoding="utf-8",
                )
        if "evaluate" in command:
            score_root = output_dir / "score" / "example_model"
            score_root.mkdir(parents=True)
            for category, case_ids in cases.items():
                path = score_root / f"BFCL_v3_{category}_score.json"
                path.write_text(
                    json.dumps(
                        {
                            "accuracy": 1.0,
                            "correct_count": len(case_ids),
                            "total_count": len(case_ids),
                        }
                    ),
                    encoding="utf-8",
                )
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        api_key=None,
        base_url="http://127.0.0.1:8000/v1",
        dry_run=False,
        model="example/model",
        output_dir=str(output_dir),
    )
    result = {"coverage": dict(config), "provenance": {}}

    _run_bfcl(args, config, result)

    assert result["execution_status"] == "complete"
    assert result["summary"] == {
        "passed": 50,
        "failed": 0,
        "total": 50,
        "completed": 50,
        "score": 1.0,
    }


def test_bfcl_stops_before_evaluation_when_generated_ids_drift(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "bfcl"
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = profile["bfcl"]
    cases = _bfcl_cases(config)
    invoked: list[str] = []

    def fake_run_command(command, **_kwargs):
        if "generate" in command:
            invoked.append("generate")
            result_root = output_dir / "result" / "example_model"
            result_root.mkdir(parents=True)
            selected = cases["simple"][:-1]
            (result_root / "BFCL_v3_simple_result.json").write_text(
                "\n".join(
                    json.dumps({"id": case_id, "result": []}) for case_id in selected
                ),
                encoding="utf-8",
            )
        elif "evaluate" in command:
            invoked.append("evaluate")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        api_key=None,
        base_url="http://127.0.0.1:8000/v1",
        dry_run=False,
        model="example/model",
        output_dir=str(output_dir),
    )
    result = {"coverage": dict(config), "provenance": {}}

    _run_bfcl(args, config, result)

    assert result["execution_status"] == "error"
    assert "did not match the fixed selection" in result["error"]
    assert invoked == ["generate"]


def test_bfcl_stops_before_evaluation_when_generation_has_inference_errors(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "bfcl"
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = profile["bfcl"]
    cases = _bfcl_cases(config)
    failed_id = cases["irrelevance"][0]
    invoked: list[str] = []

    def fake_run_command(command, **_kwargs):
        if "generate" in command:
            invoked.append("generate")
            result_root = output_dir / "result" / "example_model"
            result_root.mkdir(parents=True)
            for category, case_ids in cases.items():
                path = result_root / f"BFCL_v3_{category}_result.json"
                path.write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "id": case_id,
                                "result": (
                                    "Error during inference: 500 Server Error"
                                    if case_id == failed_id
                                    else []
                                ),
                            }
                        )
                        for case_id in case_ids
                    ),
                    encoding="utf-8",
                )
        elif "evaluate" in command:
            invoked.append("evaluate")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        api_key=None,
        base_url="http://127.0.0.1:8000/v1",
        dry_run=False,
        model="example/model",
        output_dir=str(output_dir),
    )
    result = {"coverage": dict(config), "provenance": {}}

    _run_bfcl(args, config, result)

    assert result["execution_status"] == "incomplete"
    assert result["verdict"] == "inconclusive"
    assert result["coverage"]["generation_error_count"] == 1
    assert result["coverage"]["generation_valid_count"] == 49
    assert result["summary"]["score"] is None
    assert result["summary"]["total"] == 50
    assert invoked == ["generate"]
    errors = json.loads(
        (output_dir / "bfcl-generation-errors.json").read_text(encoding="utf-8")
    )
    assert errors[0]["case_id"] == failed_id


def test_custom_dry_run_uses_realistic_qualification_matrix(tmp_path: Path) -> None:
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
    case_ids = command[command.index("--cases") + 1].split(",")
    assert len(case_ids) == 61
    assert "--exclude-cases" not in command
    assert command[command.index("--modes") + 1] == "nonstream,stream"
    selection = json.loads((output_dir / "custom-case-ids.json").read_text())
    assert selection["case_ids"] == case_ids
    assert selection["resolved_case_profile"] == "gemma4"
    assert selection["case_groups"]["model_specific"] == []
    assert selection["record_count"] == 122
    assert result["coverage"]["resolved_case_count"] == 61
    assert result["coverage"]["generic_case_count"] == 61
    assert result["coverage"]["model_specific_case_count"] == 0
    assert result["provenance"]["selection_hash"] == selection["selection_hash"]


def test_custom_dry_run_appends_model_specific_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom-kimi-k2"

    exit_code = main(
        [
            "--suite",
            "custom",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "moonshotai/Kimi-K2.6",
            "--runtime",
            "dynamo-vllm",
            "--request-contract-json",
            "{}",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    selection = json.loads((output_dir / "custom-case-ids.json").read_text())
    command = result["artifacts"]["command"]
    case_ids = command[command.index("--cases") + 1].split(",")
    assert selection["resolved_case_profile"] == "kimi_k2"
    assert selection["case_ids"] == case_ids
    assert len(selection["case_groups"]["generic"]) == 61
    assert len(selection["case_groups"]["model_specific"]) == 2
    assert selection["record_count"] == 126
    assert result["coverage"]["resolved_case_count"] == 63
    assert result["coverage"]["generic_case_count"] == 61
    assert result["coverage"]["model_specific_case_count"] == 2


def test_muse_custom_dry_run_uses_complete_sampled_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom-muse"

    exit_code = main(
        [
            "--suite",
            "custom",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "meta-models/Muse-Glimmer-30B",
            "--runtime",
            "dynamo-vllm",
            "--request-contract-json",
            '{"enabled":{},"disabled":{},"require_reasoning_exercised":true}',
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    selection = json.loads((output_dir / "custom-case-ids.json").read_text())
    command = result["artifacts"]["command"]
    case_ids = command[command.index("--cases") + 1].split(",")
    assert selection["resolved_case_profile"] == "muse_glimmer"
    assert selection["case_ids"] == case_ids
    assert len(case_ids) == 61
    assert selection["record_count"] == 122
    assert command[command.index("--temperature") + 1] == "1.0"
    assert result["coverage"]["generic_case_count"] == 0
    assert result["coverage"]["complete_profile_case_count"] == 61
    assert result["coverage"]["resolved_temperature"] == 1.0


def test_custom_normalizes_the_complete_detailed_report(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "custom"
    config, _ = _profile(DEFAULT_PROFILES, "qualification")
    custom_selection, _selection_hash = _custom_selection(
        config["custom"], "google/gemma-4-31B-it"
    )
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
                        "case_ids": custom_selection["case_ids"],
                        "modes": ["nonstream", "stream"],
                        "iterations": 1,
                    },
                    "summary": {
                        "passed": 121,
                        "failed": 1,
                        "total": 122,
                        "failure_categories": {
                            "dynamo_api": 1,
                            "infrastructure": 0,
                            "model_quality": 0,
                            "unclassified": 0,
                        },
                    },
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
    result = {"coverage": dict(config["custom"])}

    _run_custom(args, config["custom"], result)

    assert result["execution_status"] == "complete"
    assert result["verdict"] == "fail"
    assert result["summary"] == {
        "passed": 121,
        "failed": 1,
        "total": 122,
        "completed": 122,
        "score": 121 / 122,
        "failure_categories": {
            "dynamo_api": 1,
            "infrastructure": 0,
            "model_quality": 0,
            "unclassified": 0,
        },
        "failure_subcategories": {},
        "dynamo_api_failures": 1,
        "infrastructure_failures": 0,
        "model_quality_failures": 0,
        "unclassified_failures": 0,
        "dynamo_errors": 1,
        "serving_errors": 1,
    }
    assert result["coverage"]["resolved_case_profile"] == "gemma4"
    assert result["coverage"]["resolved_case_count"] == 61
    assert len(result["provenance"]["selection_hash"]) == 64


@pytest.mark.parametrize(
    ("execution_status", "failure_categories", "expected"),
    [
        (
            "complete",
            {
                "dynamo_api": 1,
                "infrastructure": 0,
                "model_quality": 0,
                "unclassified": 0,
            },
            "fail",
        ),
        (
            "complete",
            {
                "dynamo_api": 0,
                "infrastructure": 1,
                "model_quality": 0,
                "unclassified": 0,
            },
            "inconclusive",
        ),
        (
            "complete",
            {
                "dynamo_api": 0,
                "infrastructure": 0,
                "model_quality": 2,
                "unclassified": 0,
            },
            "pass",
        ),
        (
            "complete",
            {
                "dynamo_api": 0,
                "infrastructure": 0,
                "model_quality": 0,
                "unclassified": 1,
            },
            "inconclusive",
        ),
        (
            "incomplete",
            {
                "dynamo_api": 1,
                "infrastructure": 0,
                "model_quality": 0,
                "unclassified": 0,
            },
            "inconclusive",
        ),
    ],
)
def test_custom_verdict_uses_failure_ownership(
    execution_status: str, failure_categories: dict[str, int], expected: str
) -> None:
    assert _custom_verdict(execution_status, failure_categories) == expected


def test_custom_rejects_a_report_with_case_selection_drift(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "custom"
    config, _ = _profile(DEFAULT_PROFILES, "qualification")
    custom_selection, _selection_hash = _custom_selection(
        config["custom"], "google/gemma-4-31B-it"
    )
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
                        "case_ids": custom_selection["case_ids"][:-1],
                        "modes": ["nonstream", "stream"],
                        "iterations": 1,
                    },
                    "summary": {"passed": 122, "failed": 0, "total": 122},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, "")

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
    result = {"coverage": dict(config["custom"])}

    _run_custom(args, config["custom"], result)

    assert result["execution_status"] == "incomplete"
    assert result["verdict"] == "inconclusive"
    assert "did not match the fixed qualification selection" in result["error"]


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


def test_openrouter_dry_run_writes_normalized_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "openrouter"

    exit_code = main(
        [
            "--suite",
            "openrouter",
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
    assert result["suite"] == "openrouter"
    assert result["execution_status"] == "planned"
    assert result["coverage"]["generator"] == "sentinel"
    assert result["coverage"]["expected_case_count"] == 187
    assert "@sha256:" in result["provenance"]["runner_image"]
    commands = result["artifacts"]["commands"]
    image = result["provenance"]["runner_image"]
    assert [command[command.index(image) + 1] for command in commands] == [
        "dump-cases",
        "bench",
        "report",
    ]
    bench = commands[1]
    assert bench[bench.index("--benchmark") + 1] == "toolcall_schema_fuzz"
    assert bench[bench.index("--model") + 1] == "example/model"
    assert bench[bench.index("--reasoning-effort") + 1] == "low"
    solver_config = json.loads(bench[bench.index("--solver-config") + 1])
    assert solver_config == {
        "generator": "sentinel",
        "maxTokens": 2048,
        "timeoutMs": 300000,
    }
    # The endpoint travels through the environment, never through argv.
    assert all("127.0.0.1" not in argument for argument in bench)
    assert "OPENROUTER_BASE_URL" in bench and "OPENROUTER_API_KEY" in bench
    # Each run mounts its own results directory, so the report never sees
    # parquet from an earlier run in the same output directory.
    mount = bench[bench.index("-v") + 1]
    assert mount.endswith("/openrouter-results:/results")


def _openrouter_case(case_id: str, *, passes: int, failure: str | None = None) -> dict:
    return {
        "id": case_id,
        "scenario": "single",
        "constructs": ["object", "string"],
        "runs": 1,
        "passes": passes,
        "transport": 0,
        "stability": "passed_every_run" if passes else "failed_every_run",
        "failures": {} if failure is None else {failure: 1},
        "examples": {},
    }


def test_openrouter_normalizes_report_and_checks_selection(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "openrouter"
    output_dir.mkdir()
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = dict(profile["openrouter"])
    config["expected_case_count"] = 5
    case_ids = [
        f"toolcall_schema_fuzz-realistic:loose:case{index}" for index in range(5)
    ]
    report = {
        "cases": [
            _openrouter_case(case_ids[0], passes=1),
            _openrouter_case(case_ids[1], passes=1),
            _openrouter_case(case_ids[2], passes=0, failure="schema_violation"),
            _openrouter_case(case_ids[3], passes=0, failure="request_rejected"),
            {**_openrouter_case(case_ids[4], passes=0), "runs": 0, "transport": 1},
        ],
        "constructs": {},
        "scenarios": [],
        "totals": {"cases": 5, "runs": 4, "passes": 2},
    }

    def fake_run_command(command, *, log_path, **_kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "dump-cases" in command:
            log_path.write_text(
                json.dumps([{"id": case_id, "tools": []} for case_id in case_ids])
            )
        elif "report" in command:
            log_path.write_text("$ bun report\n" + json.dumps(report))
        else:
            log_path.write_text("Running toolcall_schema_fuzz ...\n")
        return subprocess.CompletedProcess([], 0, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        model="example/model",
        runtime="dynamo-vllm",
        output_dir=str(output_dir),
        api_key=None,
        dry_run=False,
    )
    result = {"coverage": dict(config), "provenance": {}, "artifacts": {}}

    _run_openrouter(args, config, result)

    assert result["execution_status"] == "incomplete"
    assert "not scored exactly once" in result["error"]
    assert result["verdict"] == "inconclusive"
    assert result["summary"] == {
        "passed": 2,
        "failed": 2,
        "total": 5,
        "completed": 4,
        "score": 0.5,
        "failure_kinds": {"request_rejected": 1, "schema_violation": 1},
        "failure_subcategories": {
            "dynamo_api.argument_encoding": 1,
            "dynamo_api.request_validation": 1,
        },
    }
    assert result["coverage"]["unscored_case_ids"] == [case_ids[4]]
    assert (output_dir / "openrouter-results").is_dir()
    assert result["artifacts"]["result_root"].endswith("openrouter-results")
    assert result["case_results"][case_ids[0]] == {"status": "pass"}
    assert result["case_results"][case_ids[3]] == {
        "status": "fail",
        "failure_kind": "request_rejected",
        "failure_subcategory": "dynamo_api.request_validation",
    }
    assert result["case_results"][case_ids[4]] == {"status": "unscored", "runs": 0}
    assert (
        json.loads((output_dir / "openrouter-case-results.json").read_text())
        == result["case_results"]
    )
    assert len(result["provenance"]["selection_hash"]) == 64
    assert json.loads((output_dir / "openrouter-case-ids.json").read_text()) == case_ids

    # A drifted case set is a contract error, not a score.
    report["cases"][0]["id"] = "toolcall_schema_fuzz-realistic:loose:renamed"
    result = {"coverage": dict(config), "provenance": {}, "artifacts": {}}
    _run_openrouter(args, config, result)
    assert result["execution_status"] == "incomplete"
    assert "missing=" in result["error"] and "unexpected=" in result["error"]


def test_openrouter_rejects_a_case_count_drift(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "openrouter"
    output_dir.mkdir()
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = profile["openrouter"]

    def fake_run_command(command, *, log_path, **_kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps([{"id": "only-one"}]))
        return subprocess.CompletedProcess([], 0, "")

    monkeypatch.setattr(
        "benchmarks.tool_calling.e2e_verifier.cli._run_command", fake_run_command
    )
    args = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        model="example/model",
        runtime="dynamo-vllm",
        output_dir=str(output_dir),
        api_key=None,
        dry_run=False,
    )
    result = {"coverage": dict(config), "provenance": {}, "artifacts": {}}

    _run_openrouter(args, config, result)

    assert result["execution_status"] == "error"
    assert "produced 1 cases, profile expects 187" in result["error"]


def test_openrouter_profile_rejects_unknown_generator() -> None:
    profile, _ = _profile(DEFAULT_PROFILES, "qualification")
    config = {**profile["openrouter"], "generator": "everything"}
    with pytest.raises(ValueError, match="generator must be one of"):
        _openrouter_config(config)


@pytest.mark.parametrize("runtime", ("vllm-serve", "sglang-serve", "dynamo-vllm"))
def test_openrouter_is_endpoint_agnostic_across_runtimes(
    tmp_path: Path, runtime: str
) -> None:
    output_dir = tmp_path / runtime

    exit_code = main(
        [
            "--suite",
            "openrouter",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "example/model",
            "--runtime",
            runtime,
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    result = json.loads((output_dir / "suite-result.json").read_text())
    assert result["runtime"] == runtime
    # The harness only sees OPENROUTER_BASE_URL; the runtime is a label.
    assert result["artifacts"]["commands"][1].count("--runtime") == 0

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded Dynamo tool-calling qualification suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..custom.model_profiles import model_case_profile
from . import CONTRACT_VERSION

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = ROOT / "profiles.json"
SUITES = ("custom", "bfcl", "tau2")
CUSTOM_SELECTION_FILE = "custom-case-ids.json"
CUSTOM_GENERIC_CASE_COUNT = 25
CUSTOM_MODES = ("nonstream", "stream")
CUSTOM_ITERATIONS = 1
CUSTOM_GENERIC_RECORD_COUNT = (
    CUSTOM_GENERIC_CASE_COUNT * len(CUSTOM_MODES) * CUSTOM_ITERATIONS
)
BFCL_SELECTION_FILE = "bfcl-case-ids.json"
BFCL_SELECTION_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/test_case_ids_to_generate.json"
)
BFCL_PREFLIGHT_SCRIPT = """
import json
from pathlib import Path

from bfcl.constant import PROMPT_PATH, TEST_FILE_MAPPING, TEST_IDS_TO_GENERATE_PATH
from bfcl.utils import load_file, sort_key

selection = json.loads(Path(TEST_IDS_TO_GENERATE_PATH).read_text(encoding="utf-8"))
errors = []
for category, case_ids in selection.items():
    if category not in TEST_FILE_MAPPING:
        errors.append(f"unknown category: {category}")
        continue
    rows = sorted(load_file(PROMPT_PATH / TEST_FILE_MAPPING[category]), key=sort_key)
    available_ids = [row["id"] for row in rows]
    missing = sorted(set(case_ids) - set(available_ids))
    if missing:
        errors.append(f"{category} missing IDs: {missing}")
    if available_ids[: len(case_ids)] != case_ids:
        errors.append(
            f"{category} IDs are not the evaluator's prefix-aligned fixed subset"
        )
if errors:
    print("BFCL selection contract failed: " + "; ".join(errors))
    raise SystemExit(2)
print(
    json.dumps(
        {
            "categories": list(selection),
            "case_count": sum(len(case_ids) for case_ids in selection.values()),
        },
        sort_keys=True,
    )
)
""".strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _profile(path: Path, name: str) -> tuple[dict[str, Any], str]:
    raw = _load_object(path)
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ValueError(f"unknown verifier profile: {name}")
    selected = profiles[name]
    if not isinstance(selected, dict):
        raise ValueError(f"profile {name} must be an object")
    if "extends" in selected:
        parent_name = str(selected["extends"])
        parent = profiles.get(parent_name)
        if not isinstance(parent, dict):
            raise ValueError(f"profile {name} extends unknown profile {parent_name}")
        selected = {
            **parent,
            **{key: value for key, value in selected.items() if key != "extends"},
        }
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return selected, hashlib.sha256(encoded).hexdigest()


def _run_command(
    argv: Sequence[str],
    *,
    timeout: float,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        tuple(argv),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed


def _base_result(
    args: argparse.Namespace,
    suite_config: Mapping[str, Any],
    profile_hash: str,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "suite": args.suite,
        "profile": args.profile,
        "model": args.model,
        "runtime": args.runtime,
        "execution_status": "planned" if args.dry_run else "running",
        "verdict": "inconclusive",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "summary": {"passed": 0, "failed": 0, "total": 0, "score": None},
        "coverage": dict(suite_config),
        "provenance": {
            "profile_hash": profile_hash,
            "profiles_file": str(args.profiles),
            "runner_image": suite_config.get("image"),
        },
        "artifacts": {},
        "error": None,
    }


def _finish(result: dict[str, Any], started: datetime) -> dict[str, Any]:
    finished = datetime.now(timezone.utc)
    result["finished_at"] = finished.isoformat()
    result["duration_seconds"] = round((finished - started).total_seconds(), 3)
    return result


def _case_id_group(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"custom qualification requires {label} case IDs")
    if not all(isinstance(case_id, str) and case_id for case_id in value):
        raise ValueError("custom qualification has an invalid case ID")
    case_ids = tuple(sorted(value))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("custom qualification has duplicate case IDs")
    return case_ids


def _custom_selection(
    config: Mapping[str, Any], model: str
) -> tuple[dict[str, Any], str]:
    generic_case_ids = _case_id_group(
        config.get("generic_cases"), label="fixed generic"
    )
    if len(generic_case_ids) != CUSTOM_GENERIC_CASE_COUNT:
        raise ValueError(
            "custom qualification requires exactly "
            f"{CUSTOM_GENERIC_CASE_COUNT} generic case IDs"
        )

    raw_model_specific = config.get("model_specific_cases")
    if not isinstance(raw_model_specific, dict):
        raise ValueError("custom qualification requires model-specific case groups")
    model_specific_groups = {
        str(profile): _case_id_group(case_ids, label=f"{profile} model-specific")
        for profile, case_ids in raw_model_specific.items()
        if isinstance(profile, str) and profile
    }
    if len(model_specific_groups) != len(raw_model_specific):
        raise ValueError("custom qualification has an invalid model-specific profile")
    for profile, profile_case_ids in model_specific_groups.items():
        overlap = sorted(set(generic_case_ids) & set(profile_case_ids))
        if overlap:
            raise ValueError(
                "custom qualification cases cannot be both generic and "
                f"{profile} model-specific: " + ", ".join(overlap)
            )

    configured_profile = str(config.get("case_profile", "auto"))
    resolved_profile = (
        model_case_profile(model)
        if configured_profile == "auto"
        else configured_profile
    )
    model_specific_case_ids = model_specific_groups.get(resolved_profile, ())
    case_ids = tuple(sorted((*generic_case_ids, *model_specific_case_ids)))

    raw_modes = config.get("modes")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError("custom qualification requires fixed modes")
    if not all(isinstance(mode, str) and mode for mode in raw_modes):
        raise ValueError("custom qualification has an invalid mode")
    modes = tuple(raw_modes)
    if len(modes) != len(set(modes)):
        raise ValueError("custom qualification has duplicate modes")
    if modes != CUSTOM_MODES:
        raise ValueError("custom qualification modes must be " + ",".join(CUSTOM_MODES))

    iterations = int(config.get("iterations") or 0)
    if iterations != CUSTOM_ITERATIONS:
        raise ValueError(
            f"custom qualification requires exactly {CUSTOM_ITERATIONS} iteration"
        )
    hash_payload = {
        "resolved_case_profile": resolved_profile,
        "case_groups": {
            "generic": generic_case_ids,
            "model_specific": model_specific_case_ids,
        },
        "modes": modes,
        "iterations": iterations,
    }
    selection_hash = hashlib.sha256(
        json.dumps(hash_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    model_specific_record_count = len(model_specific_case_ids) * len(modes) * iterations
    selection = {
        "case_profile": configured_profile,
        "resolved_case_profile": resolved_profile,
        "case_groups": {
            "generic": list(generic_case_ids),
            "model_specific": list(model_specific_case_ids),
        },
        "case_ids": list(case_ids),
        "modes": list(modes),
        "iterations": iterations,
        "generic_case_count": len(generic_case_ids),
        "model_specific_case_count": len(model_specific_case_ids),
        "generic_record_count": CUSTOM_GENERIC_RECORD_COUNT,
        "model_specific_record_count": model_specific_record_count,
        "record_count": CUSTOM_GENERIC_RECORD_COUNT + model_specific_record_count,
        "selection_hash": selection_hash,
    }
    return selection, selection_hash


def _record_custom_selection(
    output_dir: Path,
    config: Mapping[str, Any],
    result: dict[str, Any],
    model: str,
) -> tuple[dict[str, Any], Path]:
    selection, selection_hash = _custom_selection(config, model)
    selection_path = output_dir / CUSTOM_SELECTION_FILE
    _write_json(selection_path, selection)
    result["coverage"].update(
        {
            "resolved_case_count": len(selection["case_ids"]),
            "resolved_case_profile": selection["resolved_case_profile"],
            "case_groups": selection["case_groups"],
            "generic_case_count": selection["generic_case_count"],
            "model_specific_case_count": selection["model_specific_case_count"],
            "generic_record_count": selection["generic_record_count"],
            "model_specific_record_count": selection["model_specific_record_count"],
            "selection_hash": selection_hash,
        }
    )
    result.setdefault("provenance", {})["selection_hash"] = selection_hash
    return selection, selection_path


def _custom_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[str]:
    adapter = ROOT.parent / "custom_runner.py"
    custom_root = ROOT.parent / "custom"
    output_dir = Path(args.output_dir)
    command = [
        sys.executable,
        str(adapter),
        "--custom-root",
        str(custom_root),
        "--request-contract-json",
        args.request_contract_json,
        "--",
        "--site-dir",
        str(output_dir / "site"),
        "--output-root",
        str(output_dir / "raw-runs"),
        "--title",
        f"{args.model} · {args.runtime} custom qualification",
        "--base-url",
        args.base_url,
        "--allow-other-base-url",
        "--no-auth",
        "--model",
        args.model,
        "--model-slug",
        "qualification",
        "--model-label",
        f"{args.model} · {args.runtime}",
        "--case-profile",
        str(config.get("case_profile", "auto")),
        "--modes",
        ",".join(str(value) for value in (config.get("modes") or ())),
        "--iterations",
        str(config["iterations"]),
        "--concurrency",
        str(config["concurrency"]),
        "--temperature",
        str(config["temperature"]),
        "--max-tokens",
        str(config["max_tokens"]),
        "--timeout-seconds",
        str(min(float(config["request_timeout_seconds"]), args.request_timeout)),
        "--raw-chars",
        "20000",
        "--detail-chars",
        "40000",
        "--record-success-raw",
        "--ai-analysis-mode",
        "heuristic",
        "--no-root-alias",
        "--fail-on-test-failure",
    ]
    cases = tuple(str(value) for value in selection["case_ids"])
    if cases:
        command.extend(("--cases", ",".join(cases)))
    exclude_cases = tuple(str(value) for value in (config.get("exclude_cases") or ()))
    if exclude_cases:
        command.extend(("--exclude-cases", ",".join(exclude_cases)))
    return command


def _run_custom(
    args: argparse.Namespace, config: Mapping[str, Any], result: dict[str, Any]
) -> None:
    output_dir = Path(args.output_dir)
    selection, selection_path = _record_custom_selection(
        output_dir, config, result, args.model
    )
    completed = _run_command(
        _custom_command(args, config, selection),
        timeout=float(config["timeout_seconds"]),
        log_path=output_dir / "runner.log",
    )
    raw_report = (
        output_dir / "site" / "models" / "qualification" / "artifacts" / "latest.json"
    )
    if not raw_report.exists():
        result["execution_status"] = "error"
        result[
            "error"
        ] = f"custom validator exited {completed.returncode} without latest.json"
        result["artifacts"] = {
            "runner_log": str(output_dir / "runner.log"),
            "runner_returncode": completed.returncode,
            "selection": str(selection_path),
        }
        return
    payload = _load_object(raw_report)
    raw_summary = payload.get("summary") or {}
    raw_config = payload.get("config") or {}
    passed = int(raw_summary.get("passed") or 0)
    total = int(raw_summary.get("total") or 0)
    failed = int(raw_summary.get("failed") or 0)
    raw_case_ids = raw_config.get("case_ids")
    actual_case_ids = (
        tuple(sorted(raw_case_ids))
        if isinstance(raw_case_ids, list)
        and all(isinstance(case_id, str) for case_id in raw_case_ids)
        else ()
    )
    raw_modes = raw_config.get("modes")
    actual_modes = (
        tuple(raw_modes)
        if isinstance(raw_modes, list)
        and all(isinstance(mode, str) for mode in raw_modes)
        else ()
    )
    actual_iterations = int(raw_config.get("iterations") or 0)
    actual_profile = raw_config.get("case_profile")
    expected_case_ids = tuple(selection["case_ids"])
    expected_modes = tuple(selection["modes"])
    expected_iterations = int(selection["iterations"])
    expected = int(selection["record_count"])
    matrix_matches = (
        actual_case_ids == expected_case_ids
        and actual_modes == expected_modes
        and actual_iterations == expected_iterations
        and actual_profile == selection["resolved_case_profile"]
    )
    result["execution_status"] = (
        "complete" if matrix_matches and total == expected else "incomplete"
    )
    if not matrix_matches:
        result["error"] = (
            "custom report matrix did not match the fixed qualification selection: "
            f"expected_cases={list(expected_case_ids)}, "
            f"actual_cases={list(actual_case_ids)}, "
            f"expected_modes={list(expected_modes)}, actual_modes={list(actual_modes)}, "
            f"expected_iterations={expected_iterations}, "
            f"actual_iterations={actual_iterations}, "
            f"expected_profile={selection['resolved_case_profile']}, "
            f"actual_profile={actual_profile}"
        )
    elif total != expected:
        result[
            "error"
        ] = f"custom report completed {total}/{expected} fixed qualification records"
    result["verdict"] = (
        "pass"
        if result["execution_status"] == "complete" and failed == 0
        else "fail"
        if result["execution_status"] == "complete"
        else "inconclusive"
    )
    result["summary"] = {
        "passed": passed,
        "failed": failed,
        "total": expected,
        "completed": total,
        "score": passed / total if total else None,
    }
    result["coverage"].update(
        {
            "resolved_case_profile": actual_profile,
            "resolved_case_count": len(actual_case_ids),
        }
    )
    result["artifacts"] = {
        "raw_report": str(raw_report),
        "runner_log": str(output_dir / "runner.log"),
        "runner_returncode": completed.returncode,
        "selection": str(selection_path),
    }


def _docker_prefix(
    image: str,
    output_dir: Path,
    *,
    forwarded_environment: Sequence[str] = (),
    read_only_mounts: Sequence[tuple[Path, str]] = (),
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{output_dir.resolve()}:/results",
    ]
    for host_path, container_path in read_only_mounts:
        command.extend(("-v", f"{host_path.resolve()}:{container_path}:ro"))
    for name in forwarded_environment:
        command.extend(("--env", name))
    command.append(image)
    return command


def _redact_commands(
    commands: Sequence[Sequence[str]], secrets: Sequence[str | None]
) -> list[list[str]]:
    redacted = [secret for secret in secrets if secret]
    return [
        [_redact_argument(argument, redacted) for argument in command]
        for command in commands
    ]


def _redact_argument(argument: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        argument = argument.replace(secret, "<redacted>")
    return argument


def _bfcl_commands(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> list[list[str]]:
    output_dir = Path(args.output_dir)
    image = str(config["image"])
    native = str(bool(config.get("native_calling", True))).lower()
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    model_args = f"base_url={endpoint},native_calling={native}"
    cases = _bfcl_cases(config)
    categories = list(cases)
    selection_path = output_dir / BFCL_SELECTION_FILE
    _write_json(selection_path, cases)
    selection_mount = ((selection_path, BFCL_SELECTION_CONTAINER_PATH),)
    common = [
        "--model",
        args.model,
        "--test-category",
        ",".join(categories),
        "--model-mapping",
        "oai",
        "--result-dir",
        "/results",
        "--model-args",
        model_args,
    ]
    preflight = _docker_prefix(image, output_dir, read_only_mounts=selection_mount) + [
        "python",
        "-c",
        BFCL_PREFLIGHT_SCRIPT,
    ]
    generate = _docker_prefix(
        image,
        output_dir,
        forwarded_environment=("OPENAI_API_KEY",),
        read_only_mounts=selection_mount,
    ) + [
        "bfcl",
        "generate",
        *common,
        "--run-ids",
        "--num-threads",
        str(config["parallelism"]),
    ]
    evaluate = _docker_prefix(
        image, output_dir, forwarded_environment=("OPENAI_API_KEY",)
    ) + [
        "bfcl",
        "evaluate",
        "--model",
        args.model,
        "--test-category",
        ",".join(categories),
        "--model-mapping",
        "oai",
        "--result-dir",
        "/results",
        "--score-dir",
        "/results",
        "--model-args",
        model_args,
    ]
    return [preflight, generate, evaluate]


def _bfcl_cases(config: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_cases = config.get("cases")
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise ValueError("BFCL profile requires a non-empty cases mapping")
    cases: dict[str, list[str]] = {}
    seen: set[str] = set()
    for raw_category, raw_ids in raw_cases.items():
        category = str(raw_category)
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"BFCL category {category} requires fixed case IDs")
        if not all(isinstance(case_id, str) for case_id in raw_ids):
            raise ValueError(f"BFCL category {category} has a non-string case ID")
        case_ids = list(raw_ids)
        mismatched = [
            case_id for case_id in case_ids if not case_id.startswith(f"{category}_")
        ]
        if mismatched:
            raise ValueError(
                f"BFCL category {category} has mismatched case IDs: {mismatched}"
            )
        duplicates = sorted(seen.intersection(case_ids))
        if duplicates or len(set(case_ids)) != len(case_ids):
            raise ValueError(
                f"BFCL selection has duplicate case IDs: {duplicates or case_ids}"
            )
        seen.update(case_ids)
        cases[category] = case_ids
    return cases


def _bfcl_generated_ids(output_dir: Path) -> list[str]:
    generated: list[str] = []
    for path in output_dir.glob("result/*/BFCL_v3_*_result.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            case_id = payload.get("id")
            if not isinstance(case_id, str):
                raise ValueError(f"BFCL result in {path} has no string case ID")
            generated.append(case_id)
    return generated


def _bfcl_category_counts(output_dir: Path) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    prefix = "BFCL_v3_"
    suffix = "_score.json"
    for path in output_dir.rglob(f"{prefix}*{suffix}"):
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            header = json.loads(first_line)
            category = path.name[len(prefix) : -len(suffix)]
            previous_correct, previous_total = counts.get(category, (0, 0))
            counts[category] = (
                previous_correct + int(header["correct_count"]),
                previous_total + int(header["total_count"]),
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return counts


def _bfcl_counts(output_dir: Path) -> tuple[int, int]:
    counts = _bfcl_category_counts(output_dir)
    return (
        sum(correct for correct, _ in counts.values()),
        sum(total for _, total in counts.values()),
    )


def _run_bfcl(
    args: argparse.Namespace, config: Mapping[str, Any], result: dict[str, Any]
) -> None:
    output_dir = Path(args.output_dir)
    cases = _bfcl_cases(config)
    expected_ids = [case_id for case_ids in cases.values() for case_id in case_ids]
    selection_hash = hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["coverage"].update(
        {"resolved_case_count": len(expected_ids), "selection_hash": selection_hash}
    )
    result["provenance"]["selection_hash"] = selection_hash
    commands = _bfcl_commands(args, config)
    if args.dry_run:
        result["artifacts"] = {
            "commands": commands,
            "selection": str(output_dir / BFCL_SELECTION_FILE),
        }
        return
    log_parts: list[str] = []
    deadline = time.monotonic() + float(config["timeout_seconds"])
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = args.api_key or "dummy"
    phases = ("selection contract validation", "generation", "evaluation")
    for index, command in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, config["timeout_seconds"])
        completed = _run_command(
            command,
            timeout=remaining,
            log_path=output_dir / f"command-{index:02d}.log",
            env=environment,
        )
        log_parts.append(str(output_dir / f"command-{index:02d}.log"))
        if completed.returncode != 0:
            result["execution_status"] = "error"
            result["error"] = f"BFCL {phases[index]} exited {completed.returncode}"
            result["artifacts"] = {
                "runner_logs": log_parts,
                "selection": str(output_dir / BFCL_SELECTION_FILE),
            }
            return
        if phases[index] == "generation":
            try:
                generated_ids = _bfcl_generated_ids(output_dir)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                result["execution_status"] = "error"
                result["error"] = f"BFCL could not validate generated case IDs: {exc}"
                result["artifacts"] = {
                    "runner_logs": log_parts,
                    "selection": str(output_dir / BFCL_SELECTION_FILE),
                }
                return
            missing = sorted(set(expected_ids) - set(generated_ids))
            unexpected = sorted(set(generated_ids) - set(expected_ids))
            if missing or unexpected or len(generated_ids) != len(expected_ids):
                result["execution_status"] = "error"
                result["error"] = (
                    "BFCL generated case IDs did not match the fixed selection: "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"generated={len(generated_ids)}, expected={len(expected_ids)}"
                )
                result["artifacts"] = {
                    "runner_logs": log_parts,
                    "selection": str(output_dir / BFCL_SELECTION_FILE),
                }
                return
    expected_counts = {category: len(case_ids) for category, case_ids in cases.items()}
    category_counts = _bfcl_category_counts(output_dir)
    completed_counts = {
        category: total for category, (_, total) in category_counts.items()
    }
    correct = sum(correct for correct, _ in category_counts.values())
    completed_total = sum(completed_counts.values())
    total = len(expected_ids)
    score = correct / completed_total if completed_total else None
    result["execution_status"] = (
        "complete" if completed_counts == expected_counts else "incomplete"
    )
    if result["execution_status"] == "incomplete":
        result["error"] = (
            "BFCL scored counts did not match the fixed selection: "
            f"expected={expected_counts}, completed={completed_counts}"
        )
    result["verdict"] = "inconclusive"
    result["summary"] = {
        "passed": correct,
        "failed": completed_total - correct,
        "total": total,
        "completed": completed_total,
        "score": score,
    }
    result["artifacts"] = {
        "runner_logs": log_parts,
        "result_root": str(output_dir),
        "selection": str(output_dir / BFCL_SELECTION_FILE),
    }


def _tau2_commands(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> list[list[str]]:
    if not args.simulator_model or not args.simulator_base_url:
        raise ValueError("tau2 requires --simulator-model and --simulator-base-url")
    output_dir = Path(args.output_dir)
    image = str(config["image"])
    agent_args = {
        "api_base": args.base_url.rstrip("/"),
        "api_key": args.api_key or "dummy",
    }
    simulator_args = {
        "api_base": args.simulator_base_url.rstrip("/"),
        "api_key": args.simulator_api_key or "dummy",
    }
    commands: list[list[str]] = []
    for domain, task_ids in (config.get("tasks") or {}).items():
        commands.append(
            _docker_prefix(image, output_dir)
            + [
                "tau2",
                "run",
                "--domain",
                str(domain),
                "--agent-llm",
                f"openai/{args.model}",
                "--user-llm",
                f"openai/{args.simulator_model}",
                "--agent-llm-args",
                json.dumps(agent_args, sort_keys=True),
                "--user-llm-args",
                json.dumps(simulator_args, sort_keys=True),
                "--num-trials",
                str(config["trials"]),
                "--task-ids",
                *(str(task_id) for task_id in task_ids),
                "--max-concurrency",
                str(config["max_concurrency"]),
                "--seed",
                str(config["seed"]),
                "--save-to",
                f"/results/{domain}",
            ]
        )
    return commands


def _tau_rewards(output_dir: Path) -> list[float]:
    rewards: list[float] = []
    for path in output_dir.rglob("*.json"):
        try:
            payload = _load_object(path)
        except (json.JSONDecodeError, ValueError):
            continue
        items = payload.get("simulations") or payload.get("results") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            reward = item.get("reward")
            if reward is None and isinstance(item.get("reward_info"), dict):
                reward = item["reward_info"].get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
    return rewards


def _run_tau2(
    args: argparse.Namespace, config: Mapping[str, Any], result: dict[str, Any]
) -> None:
    output_dir = Path(args.output_dir)
    commands = _tau2_commands(args, config)
    if args.dry_run:
        result["artifacts"] = {
            "commands": _redact_commands(
                commands, (args.api_key, args.simulator_api_key)
            )
        }
        return
    logs: list[str] = []
    deadline = time.monotonic() + float(config["timeout_seconds"])
    for index, command in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, config["timeout_seconds"])
        completed = _run_command(
            command,
            timeout=remaining,
            log_path=output_dir / f"domain-{index:02d}.log",
        )
        logs.append(str(output_dir / f"domain-{index:02d}.log"))
        if completed.returncode != 0:
            result["execution_status"] = "error"
            result["error"] = f"tau2 domain command exited {completed.returncode}"
            result["artifacts"] = {"runner_logs": logs}
            return
    rewards = _tau_rewards(output_dir)
    expected = sum(
        len(values) for values in (config.get("tasks") or {}).values()
    ) * int(config["trials"])
    score = sum(rewards) / len(rewards) if rewards else None
    result["execution_status"] = (
        "complete" if len(rewards) == expected else "incomplete"
    )
    result["verdict"] = "inconclusive"
    result["summary"] = {
        "passed": sum(reward > 0 for reward in rewards),
        "failed": sum(reward <= 0 for reward in rewards),
        "total": expected,
        "completed": len(rewards),
        "score": score,
    }
    result["artifacts"] = {"runner_logs": logs, "result_root": str(output_dir)}


def run(args: argparse.Namespace) -> int:
    profile, profile_hash = _profile(Path(args.profiles), args.profile)
    config = profile.get(args.suite)
    if not isinstance(config, dict):
        raise ValueError(f"profile {args.profile} has no {args.suite} suite")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    result = _base_result(args, config, profile_hash, started.isoformat())
    try:
        if args.dry_run:
            if args.suite == "custom":
                _selection, selection_path = _record_custom_selection(
                    output_dir, config, result, args.model
                )
                result["artifacts"] = {
                    "command": _redact_commands(
                        [_custom_command(args, config, _selection)], (args.api_key,)
                    )[0],
                    "selection": str(selection_path),
                }
            elif args.suite == "bfcl":
                _run_bfcl(args, config, result)
            else:
                _run_tau2(args, config, result)
        elif args.suite == "custom":
            _run_custom(args, config, result)
        elif args.suite == "bfcl":
            _run_bfcl(args, config, result)
        else:
            _run_tau2(args, config, result)
    except subprocess.TimeoutExpired as exc:
        result["execution_status"] = "incomplete"
        result["error"] = f"suite exceeded {exc.timeout}s timeout"
    _finish(result, started)
    result_path = output_dir / "suite-result.json"
    _write_json(result_path, result)
    print(result_path)
    return 0 if result["execution_status"] in {"complete", "planned"} else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--profile", default="qualification")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--request-contract-json", default="{}")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--simulator-model", default=os.environ.get("TAU2_SIMULATOR_MODEL")
    )
    parser.add_argument(
        "--simulator-base-url", default=os.environ.get("TAU2_SIMULATOR_BASE_URL")
    )
    parser.add_argument(
        "--simulator-api-key", default=os.environ.get("TAU2_SIMULATOR_API_KEY")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

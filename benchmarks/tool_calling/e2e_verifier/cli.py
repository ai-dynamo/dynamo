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

from . import CONTRACT_VERSION

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = ROOT / "profiles.json"
SUITES = ("custom", "bfcl", "tau2")


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


def _custom_command(args: argparse.Namespace, config: Mapping[str, Any]) -> list[str]:
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
    cases = tuple(str(value) for value in (config.get("cases") or ()))
    if cases:
        command.extend(("--cases", ",".join(cases)))
    return command


def _run_custom(
    args: argparse.Namespace, config: Mapping[str, Any], result: dict[str, Any]
) -> None:
    output_dir = Path(args.output_dir)
    completed = _run_command(
        _custom_command(args, config),
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
        return
    payload = _load_object(raw_report)
    raw_summary = payload.get("summary") or {}
    raw_config = payload.get("config") or {}
    passed = int(raw_summary.get("passed") or 0)
    total = int(raw_summary.get("total") or 0)
    failed = int(raw_summary.get("failed") or 0)
    expected = (
        len(raw_config.get("case_ids") or ())
        * len(raw_config.get("modes") or ())
        * int(raw_config.get("iterations") or 1)
    )
    result["execution_status"] = "complete" if total == expected else "incomplete"
    result["verdict"] = (
        "pass"
        if total == expected and failed == 0
        else "fail"
        if total == expected
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
            "resolved_case_profile": raw_config.get("case_profile"),
            "resolved_case_count": len(raw_config.get("case_ids") or ()),
        }
    )
    result["artifacts"] = {
        "raw_report": str(raw_report),
        "runner_log": str(output_dir / "runner.log"),
        "runner_returncode": completed.returncode,
    }


def _docker_prefix(
    image: str, output_dir: Path, *, forwarded_environment: Sequence[str] = ()
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
    commands: list[list[str]] = []
    image = str(config["image"])
    native = str(bool(config.get("native_calling", True))).lower()
    model_args = f"base_url={args.base_url.rstrip('/')},native_calling={native}"
    all_categories: list[str] = []
    for category, limit in (config.get("categories") or {}).items():
        category_value = str(category)
        all_categories.append(category_value)
        common = [
            "--model",
            args.model,
            "--test-category",
            category_value,
            "--model-mapping",
            "oai",
            "--result-dir",
            "/results",
            "--model-args",
            model_args,
        ]
        generate = _docker_prefix(
            image, output_dir, forwarded_environment=("OPENAI_API_KEY",)
        ) + [
            "bfcl",
            "generate",
            *common,
            "--limit",
            str(limit),
            "--num-threads",
            str(config["parallelism"]),
        ]
        commands.append(generate)
    evaluate = _docker_prefix(
        image, output_dir, forwarded_environment=("OPENAI_API_KEY",)
    ) + [
        "bfcl",
        "evaluate",
        "--model",
        args.model,
        "--test-category",
        ",".join(all_categories),
        "--model-mapping",
        "oai",
        "--result-dir",
        "/results",
        "--score-dir",
        "/results",
        "--model-args",
        model_args,
        "--partial-eval",
    ]
    commands.append(evaluate)
    return commands


def _bfcl_counts(output_dir: Path) -> tuple[int, int]:
    correct = 0
    total = 0
    for path in output_dir.rglob("*_score.json"):
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            header = json.loads(first_line)
            correct += int(header["correct_count"])
            total += int(header["total_count"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return correct, total


def _run_bfcl(
    args: argparse.Namespace, config: Mapping[str, Any], result: dict[str, Any]
) -> None:
    output_dir = Path(args.output_dir)
    commands = _bfcl_commands(args, config)
    if args.dry_run:
        result["artifacts"] = {"commands": commands}
        return
    log_parts: list[str] = []
    deadline = time.monotonic() + float(config["timeout_seconds"])
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = args.api_key or "dummy"
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
            result[
                "error"
            ] = f"BFCL command {index + 1}/{len(commands)} exited {completed.returncode}"
            result["artifacts"] = {"runner_logs": log_parts}
            return
    total = sum(int(value) for value in (config.get("categories") or {}).values())
    correct, completed_total = _bfcl_counts(output_dir)
    score = correct / completed_total if completed_total else None
    result["execution_status"] = (
        "complete" if completed_total == total else "incomplete"
    )
    result["verdict"] = "inconclusive"
    result["summary"] = {
        "passed": correct,
        "failed": completed_total - correct,
        "total": total,
        "completed": completed_total,
        "score": score,
    }
    result["artifacts"] = {"runner_logs": log_parts, "result_root": str(output_dir)}


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
                result["artifacts"] = {
                    "command": _redact_commands(
                        [_custom_command(args, config)], (args.api_key,)
                    )[0]
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

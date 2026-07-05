#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce compact JSON and CSV summaries from a Harbor job directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TERMINALBENCH_DIR = Path(__file__).resolve().parent
if str(TERMINALBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINALBENCH_DIR))

from capture_terminal_task_images import validate_evidence  # noqa: E402


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CAMPAIGN_SOURCE_FIELDS = {
    "schema_version",
    "source_commit",
    "source_clean",
    "source_changed_path_count",
    "bundle_sha256",
    "source_tree_sha256",
    "eval_tree_sha256",
    "campaign_env_sha256",
    "source_file_count",
    "eval_file_count",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"run metadata {field} must be a lowercase SHA-256 digest")
    return value


def validate_campaign_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CAMPAIGN_SOURCE_FIELDS:
        raise ValueError("run metadata campaign_source fields are invalid")
    if isinstance(value["schema_version"], bool) or value["schema_version"] != 1:
        raise ValueError("run metadata campaign_source schema version is invalid")
    if (
        not isinstance(value["source_commit"], str)
        or COMMIT_RE.fullmatch(value["source_commit"]) is None
    ):
        raise ValueError("run metadata campaign_source source_commit is invalid")
    if (
        value["source_clean"] is not True
        or isinstance(value["source_changed_path_count"], bool)
        or value["source_changed_path_count"] != 0
    ):
        raise ValueError("run metadata campaign_source is not clean")
    for field in (
        "bundle_sha256",
        "source_tree_sha256",
        "eval_tree_sha256",
        "campaign_env_sha256",
    ):
        _require_sha256(value[field], f"campaign_source.{field}")
    source_count = value["source_file_count"]
    eval_count = value["eval_file_count"]
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or isinstance(eval_count, bool)
        or not isinstance(eval_count, int)
        or eval_count < 1
        or source_count != eval_count + 1
    ):
        raise ValueError("run metadata campaign_source file counts are invalid")
    return value


def validate_harbor_environment(value: Any) -> dict[str, Any]:
    expected_fields = {
        "uv_sync_check",
        "python",
        "package_count",
        "packages_sha256",
        "packages",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("run metadata Harbor environment fields are invalid")
    if value["uv_sync_check"] != "passed":
        raise ValueError("run metadata Harbor environment lock check did not pass")
    if not isinstance(value["python"], str) or not value["python"]:
        raise ValueError("run metadata Harbor Python version is invalid")
    packages = value["packages"]
    if (
        not isinstance(packages, list)
        or not packages
        or not all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(component, str) and component for component in item)
            for item in packages
        )
    ):
        raise ValueError("run metadata Harbor package inventory is invalid")
    normalized_names = [
        re.sub(r"[-_.]+", "-", package[0]).casefold() for package in packages
    ]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(
            "run metadata Harbor package inventory has duplicate normalized names"
        )
    expected_order = sorted(
        packages,
        key=lambda item: (
            re.sub(r"[-_.]+", "-", item[0]).casefold(),
            item[0],
            item[1],
        ),
    )
    if packages != expected_order:
        raise ValueError("run metadata Harbor package inventory is not canonical")
    package_count = value["package_count"]
    if (
        isinstance(package_count, bool)
        or not isinstance(package_count, int)
        or package_count != len(packages)
    ):
        raise ValueError("run metadata Harbor package count is invalid")
    expected_digest = hashlib.sha256(
        json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if value["packages_sha256"] != expected_digest:
        raise ValueError("run metadata Harbor package inventory digest mismatch")
    return value


def validate_run_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("run metadata must be a JSON object")
    if (
        isinstance(metadata.get("schema_version"), bool)
        or metadata.get("schema_version") != 2
    ):
        raise ValueError("run metadata must use schema version 2")

    campaign_source = validate_campaign_source(metadata.get("campaign_source"))
    validate_harbor_environment(metadata.get("harbor_environment"))

    wrapper = metadata.get("runtime_binding")
    wrapper_fields = {"file", "deployment_sha256", "content_sha256", "content"}
    if not isinstance(wrapper, dict) or set(wrapper) != wrapper_fields:
        raise ValueError("run metadata runtime_binding fields are invalid")
    if wrapper["file"] != "runtime-binding.json":
        raise ValueError("run metadata runtime binding file name is invalid")
    _require_sha256(wrapper["deployment_sha256"], "runtime_binding.deployment_sha256")
    _require_sha256(wrapper["content_sha256"], "runtime_binding.content_sha256")

    content = wrapper["content"]
    if not isinstance(content, dict) or set(content) != {"deployment", "evaluator"}:
        raise ValueError("run metadata runtime binding content is invalid")
    deployment = content["deployment"]
    evaluator = content["evaluator"]
    if not isinstance(deployment, dict) or not isinstance(evaluator, dict):
        raise ValueError("run metadata runtime binding objects are invalid")
    if wrapper["deployment_sha256"] != canonical_sha256(deployment):
        raise ValueError("run metadata runtime binding deployment digest mismatch")
    if wrapper["content_sha256"] != canonical_sha256(content):
        raise ValueError("run metadata runtime binding content digest mismatch")
    if evaluator != {"campaign_source": campaign_source}:
        raise ValueError(
            "run metadata campaign evaluator source identity is inconsistent"
        )
    recipe = deployment.get("recipe")
    if (
        not isinstance(recipe, dict)
        or recipe.get("source_commit") != campaign_source["source_commit"]
    ):
        raise ValueError("run metadata deployment source commit is inconsistent")

    run_spec = metadata.get("run_spec")
    if (
        not isinstance(run_spec, dict)
        or run_spec.get("runtime_deployment_sha256") != wrapper["deployment_sha256"]
    ):
        raise ValueError("run metadata run specification deployment digest mismatch")
    return metadata


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def expected_task_names(dataset_metadata: dict[str, Any], count: int) -> list[str]:
    refs = dataset_metadata.get("task_refs")
    if not isinstance(refs, list):
        raise ValueError("dataset metadata task_refs must be a list")
    names = []
    for ref in refs:
        if not isinstance(ref, dict):
            names.append(None)
            continue
        name = ref.get("name")
        org = ref.get("org")
        names.append(f"{org}/{name}" if isinstance(org, str) and org else name)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("dataset metadata contains an invalid task name")
    if len(set(names)) != len(names):
        raise ValueError("dataset metadata contains duplicate task names")
    if dataset_metadata.get("task_count") != len(names):
        raise ValueError("dataset metadata task_count does not match task_refs length")
    if count > len(names):
        raise ValueError(
            f"requested {count} tasks but dataset metadata contains {len(names)}"
        )
    return names[:count]


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_seconds(started: Any, finished: Any) -> float | None:
    start = parse_datetime(started)
    finish = parse_datetime(finished)
    if start is None or finish is None:
        return None
    return max(0.0, (finish - start).total_seconds())


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    fraction = position - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 6)


def agent_totals(result: dict[str, Any]) -> dict[str, float | int | None]:
    contexts: list[dict[str, Any]] = []
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        contexts.append(agent_result)
    else:
        for step in result.get("step_results") or []:
            if isinstance(step, dict) and isinstance(step.get("agent_result"), dict):
                contexts.append(step["agent_result"])

    totals: dict[str, float | int | None] = {
        "input_tokens": None,
        "cache_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    source_fields = {
        "input_tokens": "n_input_tokens",
        "cache_tokens": "n_cache_tokens",
        "output_tokens": "n_output_tokens",
        "cost_usd": "cost_usd",
    }
    for destination, source in source_fields.items():
        values = [
            context[source] for context in contexts if context.get(source) is not None
        ]
        if values:
            totals[destination] = sum(values)
    return totals


def pass_at_k(n_attempts: int, n_passed: int, k: int) -> float | None:
    if n_attempts < k or k < 1:
        return None
    if n_attempts - n_passed < k:
        return 1.0
    return 1.0 - math.comb(n_attempts - n_passed, k) / math.comb(n_attempts, k)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def copy_regular_file(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    if sha256_file(source) != sha256_file(destination):
        raise OSError(f"copied file hash mismatch: {destination}")


def compact_job_stats(stats: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: stats.get(key)
        for key in (
            "n_completed_trials",
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
            "n_retries",
            "n_input_tokens",
            "n_cache_tokens",
            "n_output_tokens",
            "cost_usd",
        )
        if key in stats
    }
    compact["evals"] = {
        name: {
            key: evaluation.get(key)
            for key in ("n_trials", "n_errors", "metrics", "pass_at_k")
            if key in evaluation
        }
        for name, evaluation in (stats.get("evals") or {}).items()
        if isinstance(evaluation, dict)
    }
    return compact


def summarize(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    job_dir = args.job_dir.resolve()
    job_result_path = job_dir / "result.json"
    if not job_result_path.is_file():
        raise FileNotFoundError(f"Harbor job result not found: {job_result_path}")
    job_result = load_json(job_result_path)
    dataset_metadata_path = job_dir / "dataset-metadata.json"
    dataset_metadata = load_json(dataset_metadata_path)
    expected_names = expected_task_names(dataset_metadata, args.expected_tasks)
    metadata = load_json(args.metadata) if args.metadata else None
    if metadata is not None:
        validate_run_metadata(metadata)
    task_images_path = args.task_images.resolve()
    if not task_images_path.is_file() or task_images_path.is_symlink():
        raise ValueError("task-image evidence must be a regular file")
    task_images = validate_evidence(
        load_json(task_images_path),
        dataset_metadata,
        job_dir,
        args.expected_tasks,
        args.expected_attempts,
    )

    result_paths = sorted(
        path for path in job_dir.glob("*/result.json") if path.parent.name != "summary"
    )
    raw_trials = [(path, load_json(path)) for path in result_paths]
    reward_keys = sorted(
        {
            key
            for _, result in raw_trials
            for key in ((result.get("verifier_result") or {}).get("rewards") or {})
        }
    )
    primary_reward = (
        "reward"
        if "reward" in reward_keys
        else "score"
        if "score" in reward_keys
        else reward_keys[0]
        if reward_keys
        else None
    )

    trials: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    seen_names: set[str] = set()
    for path, result in raw_trials:
        trial_name = str(result.get("trial_name") or path.parent.name)
        if trial_name in seen_names:
            duplicate_names.append(trial_name)
        seen_names.add(trial_name)

        task_name = str(result.get("task_name") or "")
        verifier = result.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        primary_value = numeric(rewards.get(primary_reward)) if primary_reward else None
        exception = result.get("exception_info") or {}
        exception_type = exception.get("exception_type")
        if exception_type:
            status = "error"
        elif primary_value is None:
            status = "no_reward"
        elif primary_value > 0:
            status = "passed"
        else:
            status = "failed"

        agent_info = result.get("agent_info") or {}
        model_info = agent_info.get("model_info") or {}
        row: dict[str, Any] = {
            "task_name": task_name,
            "trial_name": trial_name,
            "status": status,
            "primary_reward": primary_value,
            "rewards_json": json.dumps(rewards, sort_keys=True, separators=(",", ":")),
            "exception_type": exception_type,
            "exception_message": exception.get("exception_message"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "elapsed_seconds": elapsed_seconds(
                result.get("started_at"), result.get("finished_at")
            ),
            "agent": agent_info.get("name"),
            "agent_version": agent_info.get("version"),
            "model": model_info.get("name"),
            "provider": model_info.get("provider"),
            "result_path": str(path.relative_to(job_dir)),
            "result_sha256": sha256_file(path),
        }
        row.update(agent_totals(result))
        trials.append(row)

    trials.sort(key=lambda row: (row["task_name"], row["trial_name"]))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_task[trial["task_name"]].append(trial)

    tasks: list[dict[str, Any]] = []
    for task_name in sorted(by_task):
        task_trials = by_task[task_name]
        passed = sum(trial["status"] == "passed" for trial in task_trials)
        errors = sum(trial["status"] == "error" for trial in task_trials)
        missing = sum(trial["status"] == "no_reward" for trial in task_trials)
        numeric_rewards = [
            trial["primary_reward"]
            for trial in task_trials
            if trial["primary_reward"] is not None
        ]
        task: dict[str, Any] = {
            "task_name": task_name,
            "attempts": len(task_trials),
            "passed_attempts": passed,
            "failed_attempts": len(task_trials) - passed - errors - missing,
            "errored_attempts": errors,
            "no_reward_attempts": missing,
            "mean_reward_scored": (
                round(sum(numeric_rewards) / len(numeric_rewards), 8)
                if numeric_rewards
                else None
            ),
            "mean_reward_all_attempts": (
                round(sum(numeric_rewards) / len(task_trials), 8)
                if task_trials
                else None
            ),
        }
        for k in range(1, args.expected_attempts + 1):
            task[f"pass_at_{k}"] = pass_at_k(len(task_trials), passed, k)
        tasks.append(task)

    exception_counts = Counter(
        str(trial["exception_type"])
        for trial in trials
        if trial["exception_type"] is not None
    )
    status_counts = Counter(str(trial["status"]) for trial in trials)
    durations = [
        float(trial["elapsed_seconds"])
        for trial in trials
        if trial["elapsed_seconds"] is not None
    ]
    primary_values = [
        float(trial["primary_reward"])
        for trial in trials
        if trial["primary_reward"] is not None
    ]
    expected_trials = args.expected_tasks * args.expected_attempts

    stats = job_result.get("stats") or {}
    validation_errors: list[str] = []
    if len(tasks) != args.expected_tasks:
        validation_errors.append(
            f"expected {args.expected_tasks} unique tasks, found {len(tasks)}"
        )
    observed_names = set(by_task)
    expected_name_set = set(expected_names)
    missing_names = sorted(expected_name_set - observed_names)
    unexpected_names = sorted(observed_names - expected_name_set)
    if missing_names or unexpected_names:
        validation_errors.append(
            "task ID mismatch: "
            + json.dumps(
                {"missing": missing_names, "unexpected": unexpected_names},
                sort_keys=True,
            )
        )
    if len(trials) != expected_trials:
        validation_errors.append(
            f"expected {expected_trials} trial results, found {len(trials)}"
        )
    incorrect_attempts = {
        task["task_name"]: task["attempts"]
        for task in tasks
        if task["attempts"] != args.expected_attempts
    }
    if incorrect_attempts:
        validation_errors.append(
            "tasks with unexpected attempt counts: "
            + json.dumps(incorrect_attempts, sort_keys=True)
        )
    if duplicate_names:
        validation_errors.append(
            "duplicate trial names: " + ", ".join(sorted(set(duplicate_names)))
        )
    if status_counts["error"]:
        validation_errors.append(
            f"{status_counts['error']} trial(s) ended with an exception"
        )
    if status_counts["no_reward"]:
        validation_errors.append(
            f"{status_counts['no_reward']} trial(s) produced no verifier reward"
        )
    if job_result.get("n_total_trials") != expected_trials:
        validation_errors.append(
            f"job result n_total_trials={job_result.get('n_total_trials')!r}, "
            f"expected {expected_trials}"
        )
    completed = stats.get("n_completed_trials", stats.get("n_trials"))
    if completed != expected_trials:
        validation_errors.append(
            f"job result completed trials={completed!r}, expected {expected_trials}"
        )

    pass_at_k_summary: dict[str, float | None] = {}
    for k in range(1, args.expected_attempts + 1):
        values = [
            task[f"pass_at_{k}"] for task in tasks if task[f"pass_at_{k}"] is not None
        ]
        pass_at_k_summary[str(k)] = (
            round(sum(values) / len(values), 8) if values else None
        )

    token_totals: dict[str, int | float | None] = {}
    for field in ("input_tokens", "cache_tokens", "output_tokens", "cost_usd"):
        values = [
            value
            for value in (trial[field] for trial in trials)
            if isinstance(value, (int, float))
        ]
        token_totals[field] = sum(values) if values else None
    summary = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "job_dir": str(job_dir),
        "validation": {
            "strict": args.strict,
            "complete": not validation_errors,
            "errors": validation_errors,
            "expected_tasks": args.expected_tasks,
            "expected_attempts_per_task": args.expected_attempts,
            "expected_trials": expected_trials,
            "observed_tasks": len(tasks),
            "observed_trials": len(trials),
            "expected_task_names": expected_names,
        },
        "score": {
            "primary_reward_key": primary_reward,
            "reward_keys": reward_keys,
            "mean_reward_scored": (
                round(sum(primary_values) / len(primary_values), 8)
                if primary_values
                else None
            ),
            "mean_reward_all_trials": (
                round(sum(primary_values) / len(trials), 8) if trials else None
            ),
            "passed_attempts": status_counts["passed"],
            "failed_attempts": status_counts["failed"],
            "errored_attempts": status_counts["error"],
            "no_reward_attempts": status_counts["no_reward"],
            "pass_at_k": pass_at_k_summary,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "exception_counts": dict(sorted(exception_counts.items())),
        "timing_seconds": {
            "sum": round(sum(durations), 6),
            "mean": round(sum(durations) / len(durations), 6) if durations else None,
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
        },
        "token_and_cost_totals": token_totals,
        "harbor_job": {
            "id": job_result.get("id"),
            "started_at": job_result.get("started_at"),
            "finished_at": job_result.get("finished_at"),
            "n_total_trials": job_result.get("n_total_trials"),
            "stats": compact_job_stats(stats),
        },
        "input_hashes": {
            "job_result_sha256": sha256_file(job_result_path),
            "job_config_sha256": sha256_file(job_dir / "config.json"),
            "job_lock_sha256": sha256_file(job_dir / "lock.json"),
            "dataset_metadata_sha256": sha256_file(dataset_metadata_path),
            "task_images_sha256": sha256_file(task_images_path),
            "run_metadata_sha256": sha256_file(args.metadata)
            if args.metadata
            else None,
        },
        "run_metadata": metadata,
        "task_images": task_images,
        "tasks": tasks,
    }
    return summary, validation_errors, trials


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--job-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path)
    result.add_argument("--task-images", type=Path, required=True)
    result.add_argument("--expected-tasks", type=int, required=True)
    result.add_argument("--expected-attempts", type=int, required=True)
    result.add_argument("--strict", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        summary, validation_errors, trial_rows = summarize(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"summary error: {error}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_fields = [
        "task_name",
        "attempts",
        "passed_attempts",
        "failed_attempts",
        "errored_attempts",
        "no_reward_attempts",
        "mean_reward_scored",
        "mean_reward_all_attempts",
        *[f"pass_at_{k}" for k in range(1, args.expected_attempts + 1)],
    ]
    trial_fields = [
        "task_name",
        "trial_name",
        "status",
        "primary_reward",
        "rewards_json",
        "exception_type",
        "exception_message",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "agent",
        "agent_version",
        "model",
        "provider",
        "input_tokens",
        "cache_tokens",
        "output_tokens",
        "cost_usd",
        "result_path",
        "result_sha256",
    ]
    copy_regular_file(args.task_images, output_dir / "task-images.json")
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "tasks.csv", summary["tasks"], task_fields)

    write_csv(output_dir / "trials.csv", trial_rows, trial_fields)

    print(json.dumps(summary["validation"], sort_keys=True))
    if args.strict and validation_errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

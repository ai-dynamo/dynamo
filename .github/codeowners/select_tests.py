# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select pre-merge GPU test features from changed paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codeowners_match import match  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / ".github/codeowners/areas.yaml"
FRAMEWORKS = ("vllm", "sglang", "trtllm")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("areas.yaml must contain a YAML mapping")
    return value


def load_changed_files(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a JSON array of changed paths in {path}")
    return value


def area_globs(policy: dict[str, Any], label: str) -> list[str]:
    for area in policy.get("areas", []):
        if area.get("label") == label:
            return list(area.get("path_globs", []))
    raise ValueError(f"Unknown CODEOWNERS area: {label}")


def _feature_matches(
    path: str, feature: dict[str, Any], policy: dict[str, Any]
) -> bool:
    patterns = list(feature.get("paths", []))
    for area in feature.get("areas", []):
        patterns.extend(area_globs(policy, area))
    return any(match(pattern, path) for pattern in patterns)


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(match(pattern, path) for pattern in patterns)


def _is_test_file(path: str) -> bool:
    parts = path.split("/")
    return (
        Path(parts[-1]).name.startswith("test_")
        and parts[-1].endswith(".py")
        and ("tests" in parts[:-1])
    )


def validate_selection_config(
    policy: dict[str, Any],
    tracked_paths: list[str] | set[str],
    registered_markers: set[str],
) -> None:
    selection = policy.get("test_selection")
    if not isinstance(selection, dict):
        raise ValueError("areas.yaml is missing the test_selection mapping")
    errors: list[str] = []
    configured_patterns: list[str] = []

    frameworks = selection.get("frameworks", {})
    features = selection.get("features", {})
    if set(frameworks) != set(FRAMEWORKS):
        errors.append(f"frameworks must define exactly {', '.join(FRAMEWORKS)}")
    if not isinstance(features, dict):
        errors.append("features must be a mapping")
        features = {}

    for framework, patterns in frameworks.items():
        if not isinstance(patterns, list) or not patterns:
            errors.append(f"framework {framework} must have path globs")
            continue
        configured_patterns.extend(patterns)

    for name, feature in features.items():
        if not isinstance(feature, dict):
            errors.append(f"feature {name} must be a mapping")
            continue
        paths = feature.get("paths", [])
        areas = feature.get("areas", [])
        if not isinstance(paths, list) or not isinstance(areas, list):
            errors.append(f"feature {name} paths and areas must be lists")
            continue
        if not paths and not areas:
            errors.append(f"feature {name} must define paths or areas")
        configured_patterns.extend(paths)
        for area in areas:
            try:
                configured_patterns.extend(area_globs(policy, area))
            except ValueError as exc:
                errors.append(str(exc))
        feature_markers = feature.get("markers", [])
        if not isinstance(feature_markers, list) or not feature_markers:
            errors.append(f"feature {name} must define markers")
            continue
        for marker in feature_markers:
            if marker not in registered_markers:
                errors.append(f"feature {name} uses unregistered marker {marker}")

    for key in ("image_build", "smoke_only", "shared_full"):
        patterns = selection.get(key, [])
        if not isinstance(patterns, list):
            errors.append(f"{key} must be a list")
            continue
        configured_patterns.extend(patterns)

    tracked = set(tracked_paths)
    for pattern in configured_patterns:
        if not isinstance(pattern, str) or not pattern:
            errors.append(f"invalid path glob: {pattern!r}")
        elif not any(match(pattern, path) for path in tracked):
            errors.append(f"path glob matches no tracked files: {pattern}")

    if errors:
        raise ValueError("\n".join(errors))


def classify_paths(
    changed_files: list[str], policy: dict[str, Any]
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
    set[str],
    set[str],
]:
    selection = policy.get("test_selection")
    if not isinstance(selection, dict):
        raise ValueError("areas.yaml is missing the test_selection mapping")
    framework_globs = selection.get("frameworks", {})
    features = selection.get("features", {})
    if not isinstance(framework_globs, dict) or not isinstance(features, dict):
        raise ValueError("test_selection frameworks and features must be mappings")

    classifications: dict[str, str] = {}
    markers: dict[str, set[str]] = {framework: set() for framework in FRAMEWORKS}
    changed_tests: dict[str, set[str]] = {framework: set() for framework in FRAMEWORKS}
    reasons: dict[str, set[str]] = {framework: set() for framework in FRAMEWORKS}
    globally_full: set[str] = set()
    full_frameworks: set[str] = set()
    smoke_only = selection.get("smoke_only", [])
    shared_full = selection.get("shared_full", [])

    for path in sorted(set(changed_files)):
        matched_features = [
            (name, feature)
            for name, feature in features.items()
            if _feature_matches(path, feature, policy)
        ]
        if matched_features:
            classifications[path] = "feature:" + matched_features[0][0]
            framework_matches = {
                framework
                for framework in FRAMEWORKS
                if _matches_any(path, framework_globs.get(framework, []))
            }
            scope = framework_matches or set(FRAMEWORKS)
            for name, feature in matched_features:
                feature_markers = set(feature.get("markers", []))
                for framework in scope:
                    markers[framework].update(feature_markers)
                    reasons[framework].add(f"feature:{name}:{path}")
            continue

        if _matches_any(path, selection.get("image_build", [])):
            classifications[path] = "shared_full"
            globally_full.add(f"shared_full:{path}")
            continue

        if _matches_any(path, smoke_only):
            classifications[path] = "smoke_only"
            continue

        framework_matches = {
            framework
            for framework in FRAMEWORKS
            if _matches_any(path, framework_globs.get(framework, []))
        }
        if framework_matches:
            classifications[path] = "framework:" + ",".join(sorted(framework_matches))
            full_frameworks.update(framework_matches)
            for framework in framework_matches:
                reasons[framework].add(f"framework:{path}")
            continue

        if _matches_any(path, shared_full):
            classifications[path] = "shared_full"
            globally_full.add(f"shared_full:{path}")
            continue

        if _is_test_file(path):
            classifications[path] = "test_file"
            for framework in FRAMEWORKS:
                changed_tests[framework].add(path)
                reasons[framework].add(f"test_file:{path}")
            continue

        classifications[path] = "unmapped"
        globally_full.add(f"unmapped:{path}")

    return (
        classifications,
        markers,
        changed_tests,
        reasons,
        globally_full,
        full_frameworks,
    )


def _feature_marker_universe(policy: Any) -> list[str]:
    if not isinstance(policy, dict):
        return []
    selection = policy.get("test_selection", {})
    if not isinstance(selection, dict):
        return []
    features = selection.get("features", {})
    if not isinstance(features, dict):
        return []
    markers = set()
    for feature in features.values():
        if not isinstance(feature, dict):
            continue
        configured = feature.get("markers", [])
        if isinstance(configured, (list, tuple, set)):
            markers.update(marker for marker in configured if isinstance(marker, str))
    return sorted(markers)


def _full_result(
    mode: str,
    event: str,
    pr: int | None,
    reason: str,
    *,
    policy: dict[str, Any] | None = None,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    if policy is None:
        try:
            loaded_policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            loaded_policy = {}
        policy = loaded_policy if isinstance(loaded_policy, dict) else {}
    return {
        "version": 1,
        "mode": mode,
        "event": event,
        "pr": pr,
        "feature_markers": _feature_marker_universe(policy),
        "frameworks": {
            framework: {
                "decision": "full",
                "markers": [],
                "changed_test_files": [],
                "reasons": [reason],
            }
            for framework in FRAMEWORKS
        },
        "paths": {},
    }


def select_changed_files(
    changed_files: list[str],
    policy: dict[str, Any],
    *,
    event: str,
    pr: int | None,
    mode: str = "shadow",
    ref: str = "",
    pr_event: bool | None = None,
) -> dict[str, Any]:
    normalized_mode = "selective" if mode == "selective" else "shadow"
    (
        classifications,
        markers,
        changed_tests,
        reasons,
        full_reasons,
        full_frameworks,
    ) = classify_paths(changed_files, policy)

    if not changed_files:
        full_reasons.add("empty-changed-files")
    if pr_event is None:
        pr_event = is_pull_request_event(event, ref)
    if not pr_event:
        full_reasons.add("non-pr-event")
    elif pr is None:
        full_reasons.add("missing-pr-number")

    feature_markers = _feature_marker_universe(policy)
    selected_frameworks: dict[str, dict[str, Any]] = {}
    for framework in FRAMEWORKS:
        framework_reasons = set(reasons[framework])
        framework_reasons.update(full_reasons)
        decision = (
            "full" if full_reasons or framework in full_frameworks else "selective"
        )
        selected_frameworks[framework] = {
            "decision": decision,
            "markers": sorted(markers[framework]),
            "changed_test_files": sorted(changed_tests[framework]),
            "reasons": sorted(framework_reasons),
        }

    return {
        "version": 1,
        "mode": normalized_mode,
        "event": event,
        "pr": pr,
        "feature_markers": feature_markers,
        "frameworks": selected_frameworks,
        "paths": classifications,
    }


def is_pull_request_event(event: str, ref: str) -> bool:
    return event in {"pull_request", "pull_request_target"} or bool(
        re.fullmatch(r"(?:refs/heads/)?pull-request/\d+", ref)
        or re.fullmatch(r"refs/pull/\d+/(?:merge|head)", ref)
    )


def _pull_request_number(ref: str, payload: dict | None) -> int | None:
    if payload:
        pull_request = payload.get("pull_request")
        candidate = (
            pull_request.get("number")
            if isinstance(pull_request, dict)
            else payload.get("number")
        )
        if isinstance(candidate, int):
            return candidate
    match_ref = re.fullmatch(r"(?:refs/heads/)?pull-request/(\d+)", ref)
    if match_ref:
        return int(match_ref.group(1))
    match_ref = re.fullmatch(r"refs/pull/(\d+)/(?:merge|head)", ref)
    if match_ref:
        return int(match_ref.group(1))
    return None


def load_event_payload(path: str) -> dict | None:
    if not path:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GitHub event payload must be a JSON object")
    return value


def write_summary(path: Path, selection: dict[str, Any]) -> None:
    def escape_path(value: object) -> str:
        return str(value).replace("|", "\\|").replace("`", "\\`")

    def escape_reason(value: str) -> str:
        return escape_path(" ".join(value.split()))

    def count_label(count: int, noun: str) -> str:
        return f"{count} {noun}" + ("" if count == 1 else "s")

    def format_paths(values: set[str] | list[str], limit: int) -> str:
        sorted_values = sorted(values)
        shown = ", ".join(f"`{escape_path(value)}`" for value in sorted_values[:limit])
        remaining = len(sorted_values) - limit
        if remaining > 0:
            shown += f" and {remaining} more"
        return shown

    def group_sort_key(classification: str) -> tuple[int, str]:
        if classification == "shared_full":
            return (0, classification)
        if classification == "unmapped":
            return (1, classification)
        if classification == "framework" or classification.startswith("framework:"):
            return (2, classification)
        return (3, classification)

    frameworks = selection["frameworks"]
    lines = [
        f"### GPU test selection (`{selection['mode']}`)",
        "",
        "| Framework | Decision | Runs |",
        "| --- | --- | --- |",
    ]
    for framework, decision in frameworks.items():
        if decision["decision"] == "full":
            runs = "all tests"
        else:
            runs = "smoke"
            markers = decision.get("markers", [])
            if markers:
                runs += " + " + ", ".join(markers)
            changed_test_files = decision.get("changed_test_files", [])
            if changed_test_files:
                runs += (
                    f" + {count_label(len(changed_test_files), 'changed test file')}"
                )
        lines.append(f"| {framework} | {decision['decision']} | {runs} |")

    full_reasons: set[str] = set()
    path_frameworks: dict[str, set[str]] = {}
    for framework, decision in frameworks.items():
        for reason in decision.get("reasons", []):
            if reason.startswith(("feature:", "test_file:")):
                continue
            prefix, separator, reason_path = reason.partition(":")
            if (
                separator
                and reason_path
                and prefix
                in {
                    "shared_full",
                    "unmapped",
                    "framework",
                }
            ):
                if prefix == "framework":
                    path_frameworks.setdefault(reason_path, set()).add(framework)
                continue
            full_reasons.add(reason)
    if full_reasons:
        reasons = ", ".join(
            f"`{escape_reason(reason)}`" for reason in sorted(full_reasons)
        )
        lines.extend(["", f"Full because: {reasons}"])

    paths = selection.get("paths", {})
    if isinstance(paths, dict) and paths:
        grouped_paths: dict[str, list[str]] = {}
        for changed_path, classification in paths.items():
            grouped_paths.setdefault(str(classification), []).append(str(changed_path))
        lines.extend(
            [
                "",
                f"<details><summary>Changed paths ({len(paths)})</summary>",
                "",
            ]
        )
        for classification in sorted(grouped_paths, key=group_sort_key):
            changed_paths = grouped_paths[classification]
            group_label = classification
            full_annotation = ""
            if classification in {"shared_full", "unmapped"}:
                full_annotation = ", forces full"
            elif classification == "framework" or classification.startswith(
                "framework:"
            ):
                if classification.startswith("framework:"):
                    framework_names = sorted(
                        name
                        for name in classification.partition(":")[2].split(",")
                        if name
                    )
                else:
                    framework_names = sorted(
                        {
                            framework
                            for changed_path in changed_paths
                            for framework in path_frameworks.get(changed_path, set())
                        }
                    )
                    if framework_names:
                        group_label = f"framework:{','.join(framework_names)}"
                if framework_names:
                    full_annotation = f", forces full for {', '.join(framework_names)}"
                else:
                    full_annotation = ", forces full"
            lines.append(
                f"- `{escape_path(group_label)}` "
                f"({len(changed_paths)}{full_annotation}): "
                f"{format_paths(changed_paths, 20)}"
            )
        lines.extend(["", "</details>"])
    else:
        lines.extend(["", "No changed paths."])
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def write_github_output(path: Path, selection: dict[str, Any]) -> None:
    output = {key: value for key, value in selection.items() if key != "paths"}
    value = json.dumps(output, separators=(",", ":"), sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"test_selection={value}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument(
        "--mode", default=os.environ.get("DYNAMO_CI_TEST_SELECTION_MODE", "shadow")
    )
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--pr", type=int)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event = args.event or "unknown"
    normalized_mode = "selective" if args.mode == "selective" else "shadow"
    pr: int | None = args.pr
    selection: dict[str, Any]
    policy: dict[str, Any] | None = None

    try:
        changed_files = list(args.changed_file)
        if args.changed_files_file:
            changed_files.extend(load_changed_files(args.changed_files_file))
        payload = load_event_payload(os.environ.get("GITHUB_EVENT_PATH", ""))
        if pr is None:
            pr = _pull_request_number(args.ref, payload)
        policy = load_policy(args.policy)
        selection = select_changed_files(
            changed_files,
            policy,
            event=event,
            pr=pr,
            mode=normalized_mode,
            ref=args.ref,
            pr_event=is_pull_request_event(event, args.ref),
        )
        if args.summary_file:
            write_summary(args.summary_file, selection)
    except Exception as exc:
        print(f"::warning::Test selection failed closed: {exc}", file=sys.stderr)
        selection = _full_result(
            normalized_mode,
            event,
            pr,
            f"selector-error: {exc}",
            policy=policy,
            policy_path=args.policy,
        )
        if args.summary_file:
            try:
                write_summary(args.summary_file, selection)
            except OSError:
                pass

    if args.github_output:
        try:
            write_github_output(args.github_output, selection)
        except Exception as exc:
            print(
                f"::warning::Test selection output failed closed: {exc}",
                file=sys.stderr,
            )
            selection = _full_result(
                normalized_mode,
                event,
                pr,
                f"github-output-error: {exc}",
                policy=policy,
                policy_path=args.policy,
            )
            try:
                write_github_output(args.github_output, selection)
            except OSError as output_error:
                print(
                    f"::warning::Unable to write full test selection output: {output_error}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

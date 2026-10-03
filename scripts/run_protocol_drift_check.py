# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare shipped native-server revisions on a version bump or scheduled check.

Version bumps require reviewed decisions for every source-change candidate.
Scheduled runs report candidate drift without changing pins or adopting behavior.
The output is a review artifact, never evidence of runtime conformance by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from check_protocol_pins import check_pins
from generate_protocol_inventory import OUTPUT, ROOT
from protocol_drift import (
    build_report,
    commit,
    git,
    snapshot,
    tool_provenance,
    write_json,
)

STATUSES = {
    "Compatible",
    "Upstream drift",
    "Dynamo gap",
    "Intentional divergence",
    "Unsupported",
    "Unverified",
}
DECISION_KEYS = {
    "material",
    "scope_endpoints",
    "status",
    "owner",
    "rationale",
    "next_step",
    "tracking_issue",
    "decision",
    "runtime_evidence",
    "evidence",
}


def validate_decisions(report: dict[str, Any], decisions: Any) -> None:
    """Require exact candidate coverage; a stale approval cannot bless new drift."""
    if not isinstance(decisions, dict):
        raise ValueError("triage must be an object")
    for key in ("previous_upstream_commit", "candidate_upstream_commit"):
        if decisions.get(key) != report[key]:
            raise ValueError(f"triage {key} does not match report")
    expected = {item["id"] for item in report["changes"]}
    actual = decisions.get("changes")
    if not isinstance(actual, dict):
        raise ValueError("triage changes must be an object keyed by source-change ID")
    if set(actual) != expected:
        raise ValueError("triage must cover exactly the current source-change IDs")
    for item in actual.values():
        if not isinstance(item, dict) or set(item) - DECISION_KEYS:
            raise ValueError(
                "triage may not overwrite source-change identity or payload"
            )
        material = item.get("material", True)
        if not isinstance(material, bool):
            raise ValueError("triage material must be a boolean")
        if not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("owner", "rationale", "next_step")
        ):
            raise ValueError("triage requires owner, rationale and next_step")
        for key in ("tracking_issue", "decision"):
            if key in item and (
                not isinstance(item[key], str) or not item[key].strip()
            ):
                raise ValueError(f"triage {key} must be nonempty text")
        for key in ("evidence", "runtime_evidence"):
            if key not in item:
                continue
            references = item[key] if isinstance(item[key], list) else [item[key]]
            if not references or not all(
                isinstance(reference, str) and reference.strip()
                for reference in references
            ):
                raise ValueError(f"triage {key} requires nonempty evidence references")
        if not material:
            if "status" in item or "runtime_evidence" in item:
                raise ValueError("non-material assessment must not claim compatibility")
            scope = item.get("scope_endpoints")
            if (
                not isinstance(scope, list)
                or not scope
                or scope != report.get("endpoints")
            ):
                raise ValueError(
                    "non-material assessment requires the exact report endpoint scope"
                )
            if not item.get("decision") or not item.get("evidence"):
                raise ValueError(
                    "non-material assessment requires a no-action decision and source evidence"
                )
            continue
        if "scope_endpoints" in item:
            raise ValueError("scope_endpoints belongs only to non-material assessments")
        if not isinstance(item.get("status"), str) or item["status"] not in STATUSES:
            raise ValueError("unknown compatibility status in triage")
        if not item.get("tracking_issue") and not item.get("decision"):
            raise ValueError(
                "triage requires a tracking issue or explicit no-action decision"
            )
        if item["status"] == "Compatible" and not item.get("runtime_evidence"):
            raise ValueError("Compatible status requires runtime conformance evidence")
        if item["status"] in {
            "Upstream drift",
            "Dynamo gap",
            "Unverified",
        } and not item.get("tracking_issue"):
            raise ValueError(
                "unresolved compatibility changes require a tracking issue"
            )
        if item["status"] in {"Intentional divergence", "Unsupported"} and not item.get(
            "evidence"
        ):
            raise ValueError(
                "divergent or unsupported behavior requires implementation/test evidence"
            )


def apply_decisions(report: dict[str, Any], decisions: Any) -> None:
    """Validate all decisions before changing any report evidence or assessment."""
    validate_decisions(report, decisions)
    for item in report["changes"]:
        decision = decisions["changes"][item["id"]]
        if decision.get("material") is False:
            # A source-only finding is not a seventh compatibility status, and
            # must not retain the extractor's placeholder Unverified claim.
            item.pop("status", None)
            item.pop("runtime_evidence", None)
        item.update(decision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--dynamo-repo", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline-dynamo", help="immutable Dynamo commit before the version bump"
    )
    mode.add_argument(
        "--upstream-candidate", help="immutable upstream candidate for periodic checks"
    )
    parser.add_argument("--require-triage", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; preserve previous reports")
    relative_pins = OUTPUT.relative_to(ROOT) / "vllm_pins.json"
    # Hash the same bytes we parse, including untracked pins and decision files.
    # A tracked Git diff alone cannot identify these inputs during initial rollout.
    pins_source = (args.dynamo_repo / relative_pins).read_bytes()
    context_source = (args.dynamo_repo / "container/context.yaml").read_bytes()
    pins = json.loads(pins_source)
    context = yaml.safe_load(context_source)
    check_pins(context, pins)
    provenance = {
        **tool_provenance(
            "protocol_drift.py",
            "run_protocol_drift_check.py",
            "check_protocol_pins.py",
            "generate_protocol_inventory.py",
        ),
        "pyyaml_version": yaml.__version__,
        "current_inputs_sha256": {
            "container/context.yaml": hashlib.sha256(context_source).hexdigest(),
            str(relative_pins): hashlib.sha256(pins_source).hexdigest(),
        },
        "baseline_dynamo_commit": None,
        "baseline_inputs_sha256": {},
    }
    current = {
        platform: pin["commit"]
        for pin in pins["versions"]
        for platform in pin["platforms"]
    }
    pairs: dict[tuple[str, str], list[str]] = {}
    if args.baseline_dynamo:
        baseline = commit(args.dynamo_repo, args.baseline_dynamo)
        old_context_source = git(
            args.dynamo_repo, "show", f"{baseline}:container/context.yaml"
        )
        old_context = yaml.safe_load(old_context_source)
        provenance["baseline_dynamo_commit"] = baseline
        provenance["baseline_inputs_sha256"] = {
            "container/context.yaml": hashlib.sha256(
                old_context_source.encode()
            ).hexdigest(),
            str(relative_pins): None,
        }
        old_pins_path = f"{baseline}:{relative_pins}"
        paths = git(
            args.dynamo_repo,
            "ls-tree",
            "--name-only",
            baseline,
            "--",
            str(relative_pins),
        ).splitlines()
        if paths:
            old_pins_source = git(args.dynamo_repo, "show", old_pins_path)
            old_pins = json.loads(old_pins_source)
            provenance["baseline_inputs_sha256"][str(relative_pins)] = hashlib.sha256(
                old_pins_source.encode()
            ).hexdigest()
        else:
            # Initial rollout: resolve the preceding container release tags once,
            # then record their full SHAs in the report. No mutable refs in reports.
            old_versions = {}
            for platform, config in old_context["vllm"].items():
                if isinstance(config, dict) and "runtime_image_tag" in config:
                    version = (
                        config["runtime_image_tag"]
                        .removeprefix("v")
                        .split("-ubuntu")[0]
                    )
                    old_versions.setdefault(version, []).append(platform)
            old_pins = {
                "versions": [
                    {
                        "version": version,
                        "platforms": platforms,
                        "commit": git(
                            args.upstream_repo,
                            "rev-parse",
                            "--verify",
                            f"refs/tags/v{version}^{{commit}}",
                        ).strip(),
                    }
                    for version, platforms in sorted(old_versions.items())
                ]
            }
        check_pins(old_context, old_pins)
        old = {
            platform: pin["commit"]
            for pin in old_pins["versions"]
            for platform in pin["platforms"]
        }
        for platform, candidate in current.items():
            if platform not in old:
                raise ValueError(
                    f"new platform {platform} needs an explicitly reviewed compatibility baseline"
                )
            pairs.setdefault((old[platform], candidate), []).append(platform)
    else:
        candidate = commit(args.upstream_repo, args.upstream_candidate)
        for platform, previous in current.items():
            pairs.setdefault((previous, candidate), []).append(platform)

    cache = {}
    needs_review = False
    summaries = []
    dynamo_sha = git(args.dynamo_repo, "rev-parse", "HEAD").strip()
    extractor_sha = hashlib.sha256(
        (Path(__file__).with_name("protocol_drift.py")).read_bytes()
    ).hexdigest()
    for (previous, candidate), platforms in sorted(pairs.items()):
        for revision in (previous, candidate):
            if revision not in cache:
                cache[revision] = snapshot(args.upstream_repo, revision, "vllm")
                write_json(
                    args.output_dir / f"inventory-{revision}.json", cache[revision]
                )
        report = build_report(
            cache[previous], cache[candidate], dynamo_sha, extractor_sha
        )
        report["platforms"] = platforms
        report["check_driver_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        report["dynamo_tracked_diff_sha256"] = hashlib.sha256(
            git(args.dynamo_repo, "diff", "--binary", "HEAD", "--").encode()
        ).hexdigest()
        decision_path = (
            args.dynamo_repo
            / OUTPUT.relative_to(ROOT)
            / "decisions"
            / f"{previous}-{candidate}.json"
        )
        decision_source = decision_path.read_bytes() if decision_path.exists() else None
        report["provenance"] = {
            **provenance,
            "decision_input": {
                "path": str(decision_path.relative_to(args.dynamo_repo)),
                "sha256": (
                    hashlib.sha256(decision_source).hexdigest()
                    if decision_source is not None
                    else None
                ),
            },
        }
        if report["changes"]:
            if decision_source is not None:
                try:
                    decisions = json.loads(decision_source)
                    apply_decisions(report, decisions)
                except ValueError as error:
                    needs_review = True
                    report["triage"] = f"invalid decision file: {error}"
                else:
                    report[
                        "triage"
                    ] = "reviewed decision file (approval requires code review)"
            else:
                needs_review = True
                report[
                    "triage"
                ] = "missing: review candidates and commit a decision file"
        else:
            report[
                "triage"
            ] = "no source changes; runtime compatibility is not inferred"
        name = f"{previous}-{candidate}.json"
        write_json(args.output_dir / name, report)
        summaries.append(
            f"- {', '.join(platforms)}: `{previous}` → `{candidate}`; {len(report['changes'])} candidates; {report['triage']}; report `{name}`."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.md").write_text(
        "# vLLM protocol drift review\n\n" + "\n".join(summaries) + "\n"
    )
    print("\n".join(summaries))
    return 1 if needs_review and args.require_triage else 0


if __name__ == "__main__":
    raise SystemExit(main())

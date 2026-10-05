# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render source-drift evidence as a developer review, without inferring parity.

This module only presents reports produced by the extractor/triage driver. It
does not classify changes, validate decisions, or alter the gate's exit status.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import quote, urlsplit


def text(value: Any) -> str:
    """Keep upstream declarations and decision prose inert in Markdown."""
    escaped = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+|~])", r"\\\1", escaped).replace("\n", " ")


def reference(value: str) -> str:
    """Link explicit HTTP(S) evidence; retain other references as literal text.

    Relative evidence paths belong to the recorded decision, not necessarily the
    report directory, so manufacturing a relative link would be misleading.
    """
    try:
        parsed = urlsplit(value)
    except ValueError:
        return text(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"[{text(value)}](<{quote(value, safe=':/?#@!$&+=,;%~_-.*')}>)"
    return text(value)


def payload(item: dict[str, Any], side: str) -> str:
    if side not in item:
        return "Not present.\n"
    value = json.dumps(item[side], indent=2, sort_keys=True, ensure_ascii=True)
    # Values may contain Markdown fences. Never allow upstream text to terminate
    # its code block and inject headings, links, or HTML into the review.
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    block = f"{fence}json\n{value}\n{fence}\n"
    if len(value) > 1500:
        return (
            "<details>\n<summary>Full source payload (expand)</summary>\n\n"
            + block
            + "\n</details>\n"
        )
    return block


def location(item: dict[str, Any]) -> str:
    path = item["path"]
    return " / ".join(path[1:]) or path[0]


def assessment(item: dict[str, Any]) -> str:
    if item.get("material") is False:
        return "Non-material for recorded scope"
    return item.get("status", "Unverified")


def change_detail(item: dict[str, Any], report: dict[str, Any], anchor: str) -> str:
    path = item["path"][0]
    previous = report["previous_upstream_commit"]
    candidate = report["candidate_upstream_commit"]
    lines = [
        f'<a id="{anchor}"></a>',
        "",
        f"### {text(item['kind'].capitalize())}: {text(location(item))}",
        "",
        f"Change ID: {text(item['id'])}",
        "",
        f"Source: {text(path)}",
        "",
    ]
    source_links = []
    # An added/removed declaration may still have a file on both sides. Link
    # only the side known to contain the declaration, without guessing lines.
    for label, revision, side in (
        ("Before source", previous, "before"),
        ("After source", candidate, "after"),
    ):
        if side in item:
            url = f"https://github.com/vllm-project/vllm/blob/{revision}/{quote(path, safe='/')}"
            source_links.append(f"[{label}]({url})")
    lines.extend(
        [
            " · ".join(source_links),
            "",
            "#### Before",
            "",
            payload(item, "before"),
            "#### After",
            "",
            payload(item, "after"),
        ]
    )
    if item.get("reachable_consumers"):
        lines.extend(
            [
                "<details>",
                "<summary>Reachable consumers (expand)</summary>",
                "",
                "These roots reference this declaration (or module for module-level changes); "
                "reachability is not proof of behavioral impact.",
                "",
            ]
        )
        lines.extend(f"- {text(consumer)}" for consumer in item["reachable_consumers"])
        lines.append("")
        lines.extend(["</details>", ""])
    if "methods" in item["path"] or "functions" in item["path"]:
        lines.extend(
            [
                "Executable bodies are represented by semantic AST hashes. Inspect the "
                "source links for the code change; a changed hash does not prove an API mismatch.",
                "",
            ]
        )
    lines.extend(
        [
            "#### Disposition and follow-up",
            "",
            f"- Assessment: {text(assessment(item))}",
        ]
    )
    for key, label in (
        ("owner", "Owner"),
        ("rationale", "Rationale"),
        ("decision", "Decision"),
        ("next_step", "Next action"),
    ):
        lines.append(f"- {label}: {text(item.get(key, 'Not recorded'))}")
    if "scope_endpoints" in item:
        lines.append(f"- Recorded scope: {text(', '.join(item['scope_endpoints']))}")
    for key, label in (
        ("tracking_issue", "Tracking issue"),
        ("evidence", "Supporting evidence"),
        ("runtime_evidence", "Runtime evidence"),
    ):
        values = item.get(key, [])
        if isinstance(values, str):
            values = [values]
        lines.append(
            f"- {label}: "
            + (
                "; ".join(reference(value) for value in values)
                if values
                else "Not recorded"
            )
        )
    return "\n".join(lines) + "\n"


def render_review(reports: list[tuple[str, dict[str, Any]]], gate: str) -> str:
    """Render all comparison pairs; gate is 'triage', 'drift', or 'none'.

    JSON remains the complete machine-readable record. This presentation keeps
    every candidate and payload, including missing-vs-null and unreviewed cases.
    Decision prose is reproduced as recorded, not endorsed as verified evidence.
    """
    if gate not in {"triage", "drift", "none"}:
        raise ValueError(f"unknown review gate: {gate}")
    count = sum(len(report["changes"]) for _, report in reports)
    complete = all(
        report.get("triage_complete", not report["changes"]) for _, report in reports
    )
    blocked = (gate == "triage" and not complete) or (gate == "drift" and count > 0)
    coverage_required = any(
        report.get("require_complete_coverage") for _, report in reports
    )
    coverage_complete = all(
        all(
            value and value["complete"]
            for value in report.get("dependency_coverage", {}).values()
        )
        and bool(report.get("dependency_coverage"))
        for _, report in reports
    )
    blocked |= coverage_required and not coverage_complete
    gate_label = (
        "Not enforced"
        if gate == "none" and not coverage_required
        else ("FAIL" if blocked else "PASS")
    )
    lines = [
        "# vLLM protocol drift review",
        "",
        f"{count} source-change candidates across {len(reports)} comparison pairs.",
        "",
        f"**Command gate: {gate_label}** ({gate}). "
        f"**Triage coverage: {'complete' if complete else 'incomplete'}**. Revision pairs are evaluated separately.",
        "",
        f"**Static dependency coverage: {'complete within stated scope' if coverage_complete else 'incomplete or unavailable'}**. "
        f"Coverage gate: {'required' if coverage_required else 'not enforced'}.",
        "",
        "A passing triage gate means valid dispositions were recorded, not that "
        "humans approved them or runtime compatibility was proven. Tracked unresolved "
        "gaps can pass triage. An unenforced gate can exit zero with incomplete triage.",
        "",
        "## Comparisons",
        "",
        "| Comparison | Platforms | Candidates | Triage |",
        "|---|---|---:|---|",
    ]
    for index, (_, report) in enumerate(reports, 1):
        previous = report["previous_upstream_commit"][:12]
        candidate = report["candidate_upstream_commit"][:12]
        platforms = text(", ".join(report.get("platforms", [])) or "Not specified")
        triage = (
            "Complete"
            if report.get("triage_complete", not report["changes"])
            else "Incomplete"
        )
        lines.append(
            f"| [{previous} → {candidate}](#comparison-{index}) | {platforms} | "
            f"{len(report['changes'])} | {triage} |"
        )
    lines.extend(
        [
            "",
            "## Review procedure",
            "",
            "1. Inspect each change and its pinned upstream source. Decide whether "
            "it affects the supported endpoint/behavior scope.",
            "2. Record ownership, rationale, evidence, and the next action in the "
            "decision file identified below; track unresolved gaps.",
            "3. Run affected conformance probes where needed. Rerun the same "
            "comparison into a fresh output directory with `--require-triage` "
            "using the version-bump driver.",
            "",
            "Evidence references below are supplied by decision authors; "
            "the tool validates their structure, not their contents or availability.",
            "",
        ]
    )
    for index, (filename, report) in enumerate(reports, 1):
        previous = report["previous_upstream_commit"]
        candidate = report["candidate_upstream_commit"]
        lines.extend(
            [
                f'<a id="comparison-{index}"></a>',
                "",
                f"## Comparison {index}",
                "",
                f"Upstream: `{previous}` → `{candidate}`",
                "",
                f"Dynamo checkout commit: `{report['dynamo_commit']}`",
                "",
                f"Generated: {text(report['generated_at'])}",
                "",
                f"Endpoint scope: {text(', '.join(report['endpoints']))}",
                "",
                f"[Full JSON evidence]({quote(filename, safe='')}) · "
                f"[Upstream comparison](https://github.com/vllm-project/vllm/compare/{previous}...{candidate})",
                "",
                f"Triage: {text(report.get('triage', 'Not evaluated by standalone extraction'))}",
                "",
            ]
        )
        decision = report.get("provenance", {}).get("decision_input")
        if decision:
            lines.extend(
                [
                    f"Decision file (relative to Dynamo checkout): {text(decision['path'])}",
                    "",
                    f"Decision input SHA256: {text(decision['sha256'] or 'Not present')}",
                    "",
                ]
            )
        if not report["changes"]:
            lines.extend(
                [
                    "No source changes detected in scanned modules. Runtime compatibility is not inferred.",
                    "",
                ]
            )
        else:
            counts = Counter(assessment(item) for item in report["changes"])
            lines.extend(
                [
                    "Recorded assessments: "
                    + "; ".join(
                        f"{text(status)}: {total}"
                        for status, total in sorted(counts.items())
                    ),
                    "",
                    "| Change | Location | Assessment |",
                    "|---|---|---|",
                ]
            )
            for number, item in enumerate(report["changes"], 1):
                anchor = f"comparison-{index}-change-{number}"
                lines.append(
                    f"| [{number}. {text(item['kind'])}](#{anchor}) | "
                    f"{text(item['path'][0])}: {text(location(item))} | {text(assessment(item))} |"
                )
            lines.append("")
            for number, item in enumerate(report["changes"], 1):
                lines.append(
                    change_detail(item, report, f"comparison-{index}-change-{number}")
                )
        lines.extend(["### Coverage and provenance", ""])
        for side, coverage in report.get("dependency_coverage", {}).items():
            if coverage is None:
                lines.extend(
                    [f"{side.capitalize()}: dependency coverage unavailable.", ""]
                )
                continue
            lines.extend(
                [
                    f"#### {side.capitalize()} dependency coverage",
                    "",
                    text(coverage["scope"]),
                    "",
                    f"{len(coverage['module_consumers'])} reachable modules; "
                    f"{len(coverage['unresolved'])} unresolved references.",
                    "",
                ]
            )
            if coverage["unresolved"]:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Unresolved dependencies and affected roots (expand)</summary>",
                        "",
                    ]
                )
            for problem in coverage["unresolved"]:
                lines.append(
                    f"- {text(problem['source'])}: {text(problem['symbol'])} — {text(problem['reason'])}"
                )
                lines.extend(f"  - Consumer: {text(root)}" for root in problem["roots"])
            if coverage["unresolved"]:
                lines.extend(["", "</details>", ""])
            lines.extend(
                [
                    "",
                    "Primitive leaves (implementations excluded): "
                    + text(", ".join(coverage["primitive_leaves"]) or "None"),
                    "",
                ]
            )
        lines.extend(f"- {text(limit)}" for limit in report["limitations"])
        lines.extend(
            [
                "",
                "Source snapshots, tool/interpreter hashes, and input hashes are recorded "
                "in the linked JSON evidence. Source coverage is bounded; this is not a "
                "complete dependency/schema resolver or a runtime conformance result.",
                "",
            ]
        )
    return "\n".join(lines)

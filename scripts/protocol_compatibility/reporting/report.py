# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the direct assessment without requiring developers to audit inventories."""

from __future__ import annotations

import html
import json
from collections import Counter
from typing import Any
from urllib.parse import quote, urlsplit


def code(value: Any) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return "<code>" + html.escape(text).replace("\n", " ") + "</code>"


def text(value: Any) -> str:
    return (
        html.escape(str(value))
        .replace("\n", " ")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("*", "&#42;")
        .replace("_", "&#95;")
        .replace("`", "&#96;")
    )


def fact(item: dict[str, Any], side: str) -> Any:
    value = item[side]
    aspect = item["aspect"]
    if isinstance(value, dict):
        if aspect == "handling" and side == "native":
            return value  # Show the native declaration, not its inapplicable handling slot.
        if aspect == "input_names":
            return value.get("wire_names", value)
        if aspect in value:
            return value[aspect]
    return value


def evidence_link(value: str, label: str | None = None) -> str:
    """Link normal evidence references, never executable or protocol-relative URLs."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"", "http", "https"} or value.startswith("//"):
        return code(value)
    target = quote(value, safe=":/?#=&%+@~.-_")
    return f"[{text(label or value)}]({target})"


def type_summary(value: dict[str, Any] | None, depth: int = 0) -> str:
    if value is None:
        return "unresolved"
    if depth > 3:
        return "nested schema (expand full facts)"
    if "any_of" in value:
        return " | ".join(type_summary(member, depth + 1) for member in value["any_of"])
    if "enum" in value:
        return "one of " + json.dumps(value["enum"], ensure_ascii=False)
    if value.get("type") == "array":
        return "array of " + type_summary(value.get("items"), depth + 1)
    if "properties" in value:
        names = sorted(value["properties"])
        return (
            "object {"
            + ", ".join(names[:8])
            + (f", … {len(names)} properties" if len(names) > 8 else "")
            + "}"
        )
    if "additional_properties" in value:
        return "map of " + type_summary(value["additional_properties"], depth + 1)
    return str(value.get("type", "unresolved"))


def fact_summary(value: Any, aspect: str) -> str:
    if value is None:
        return "unresolved / absent observation"
    if isinstance(value, dict) and "nested_inputs" in value:
        locations = [
            "/"
            + "/".join(
                segment.replace("~", "~0").replace("/", "~1")
                for segment in item["wire_path"]
            )
            for item in value["nested_inputs"]
        ]
        return "nested public locations (JSON Pointer): " + ", ".join(locations)
    if isinstance(value, dict) and "wire_type" in value:
        names = ", ".join(value.get("wire_names", [])) or "unresolved input names"
        return f'{names}: {type_summary(value["wire_type"])}; required={value.get("required")}; nullable={value.get("nullable")}'
    if isinstance(value, dict) and "effects" in value:
        effects = ", ".join(value["effects"]) or "none identified"
        stages = [
            condition.get("stage", condition.get("function", "conditional"))
            for condition in value.get("conditions", [])
        ]
        return (
            f'effects: {effects}; coverage: {"complete" if value.get("complete") else "incomplete"}'
            + ("; stages: " + ", ".join(dict.fromkeys(stages)) if stages else "")
        )
    if isinstance(value, dict) and aspect == "wire_type":
        return type_summary(value)
    if isinstance(value, dict) and "methods" in value:
        return (
            f'{len(value["methods"])} selected implementation bodies; affected scope: '
            + ", ".join(value.get("affected_fields", []))
        )
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return (
        rendered if len(rendered) <= 180 else rendered[:177] + "… (expand full facts)"
    )


def full_facts(item: dict[str, Any]) -> list[str]:
    payload = {
        "native": item["native"],
        "dynamo": item["dynamo"],
        "identity": item["identity"],
        "fingerprint": item["fingerprint"],
    }
    return [
        "",
        "<details>",
        "<summary>Full observed facts and decision identity</summary>",
        "",
        "<pre><code>"
        + html.escape(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        + "</code></pre>",
        "",
        "</details>",
        "",
    ]


def render_assessment(report: dict[str, Any]) -> str:
    gates = report["gates"]
    counts = Counter(item["category"] for item in report["findings"])
    investigation = report["investigation"]
    lines = [
        "# Dynamo–vLLM compatibility assessment",
        "",
        f'Dynamo: {code(report["revisions"]["dynamo"])}  ',
        f'vLLM: {code(report["revisions"]["vllm"])}',
        "",
        "## Scope and gate results",
        "",
        f'Scope: {code(report["scope"])}',
        "",
        "Stage 1 compares declared request contracts. Stage 2 requires separate behavioral tests.",
        "Source investigation is not implemented in this contract-only tool.",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f'| Contract extraction | {text(gates["extraction"])} |',
        f'| Contract review | {text(gates["review"])} |',
        f'| Contract policy | {text(gates["release"]["status"])} |',
        f'| Behavioral conformance | {text(report["behavioral_conformance"]["status"])} |',
        f'| Source investigation (advisory) | {text(investigation["status"])} |',
        f'| Contract CLI exit code | {report["exit_code"]} |',
        "",
        "Reviewed, handled, and zero differences do **not** establish runtime parity or release approval.",
        "",
        f'Declared contract differences: {counts["compatibility"]}. '
        f'Contract coverage gaps: {counts["coverage"]}. '
        f'Dynamo-only inputs: {counts["dynamo_specific"]}.',
        f'Contract review actions: {len(gates["pending"])}. '
        f'Lost contract coverage: {len(gates["lost_coverage"])}.',
        "",
    ]
    if report["previous_revisions"]:
        lines.extend(
            [
                f'Previous assessment: {code(report["previous_revisions"])}',
                "",
                "Changes may originate in Dynamo, upstream, or extraction coverage; upstream causality is not assumed.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Initial baseline: “new” means first observed here, not necessarily newly introduced upstream.",
                "",
            ]
        )
    sections = [
        (
            "Stage 1: new declared contract differences",
            lambda item: item["lifecycle"] == "new"
            and item["category"] == "compatibility",
        ),
        (
            "Stage 1: changed declared contract differences",
            lambda item: item["lifecycle"] == "changed"
            and item["category"] == "compatibility",
        ),
        (
            "Stage 1: existing declared contract differences",
            lambda item: item["lifecycle"] == "unchanged"
            and item["category"] == "compatibility",
        ),
        (
            "Stage 1: contract coverage gaps",
            lambda item: item["category"] == "coverage",
        ),
        (
            "Dynamo-specific inputs (not automatically defects)",
            lambda item: item["category"] == "dynamo_specific",
        ),
    ]
    for title, predicate in sections:
        selected = [item for item in report["findings"] if predicate(item)]
        lines.extend([f"## {title}", ""])
        if not selected:
            lines.extend(["None observed within this assessment's coverage.", ""])
        for item in selected:
            lines.extend(
                [
                    f'### {code(item["endpoint"])} · {code(item["path"])} · {text(item["aspect"])}',
                    "",
                    text(item["observation"]),
                    "",
                    f'- Lifecycle: {text(item["lifecycle"])}; decision: {text(item["decision_status"])}.',
                    f'- Native: {text(fact_summary(fact(item, "native"), item["aspect"]))}',
                    f'- Dynamo: {text(fact_summary(fact(item, "dynamo"), item["aspect"]))}',
                    f'- Next action: {text(item["next_action"])}',
                ]
            )
            decision = item.get("decision")
            if decision:
                lines.extend(
                    [
                        f'- Recorded disposition: {text(decision["disposition"])}; owner: {text(decision["owner"])}.',
                        f'- Rationale: {text(decision["rationale"])}',
                        f'- Tracking: {evidence_link(decision["tracking"]) if decision.get("tracking") else "explicit no-action"}',
                        f'- Current revision/config-matched runtime evidence: {len(decision.get("current_runtime_evidence", []))} record(s).',
                    ]
                )
                for evidence in decision["evidence"]:
                    lines.append(
                        f'- Evidence ({text(evidence["kind"])}; historical unless applicable): {evidence_link(evidence["url"])}'
                    )
            unique_sources = {
                json.dumps(source, sort_keys=True): source
                for source in item.get("sources", [])
            }
            for source in unique_sources.values():
                path = source.get("path", "")
                target = source.get("target")
                if (
                    target in {"vllm", "dynamo"}
                    and path
                    and not path.startswith("crate:")
                ):
                    repo = (
                        "vllm-project/vllm" if target == "vllm" else "ai-dynamo/dynamo"
                    )
                    url = f'https://github.com/{repo}/blob/{report["revisions"][target]}/{quote(path, safe="/")}'
                    lines.append(
                        f'- Source: [{text(path)}]({url}) {code(source.get("symbol", ""))}'
                    )
                else:
                    lines.append(f"- Source: {code(source)}")
            lines.extend(full_facts(item))
            lines.append("")
    lines.extend(["## Resolved or no longer assessable", ""])
    for item in report["retired_findings"]:
        lines.append(
            f'- {code(item["identity"])}: {text(item["lifecycle"])}. '
            + (
                "Restore coverage; this is not a resolution."
                if item["lifecycle"] == "no_longer_assessable"
                else "Difference no longer observed with retained coverage."
            )
        )
    if not report["retired_findings"]:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Stage 2: behavioral conformance",
            "",
            "Not assessed. No servers or behavioral tests were executed by this command.",
            "Use a separately scoped test suite with exact source/model/configuration evidence.",
            "Contract success and source notes cannot establish behavioral acceptance.",
            "",
            "## History migration",
            "",
            (
                "Legacy combined observations were mapped to contract history or retained as "
                "investigation evidence in report.json; legacy approvals were not carried forward."
                if report.get("history_migration")
                else "No legacy history migration in this run."
            ),
            "",
            "## Artifacts and next steps",
            "",
            "1. Restore contract coverage; fix declared differences or record scoped exceptions.",
            "2. Reassess with contract decisions and policy; tracked gaps are not fixes.",
            "3. Separately run the agreed behavioral suite, including schema-identical behavior.",
            "4. Consult optional source notes to investigate failures, not to approve compatibility.",
            "",
            "[Machine-readable assessment](report.json), [native inventory](native-contract.json),",
            "[Dynamo inventory](dynamo-contract.json), [native source snapshot](native-source.json).",
            "",
        ]
    )
    return "\n".join(lines)

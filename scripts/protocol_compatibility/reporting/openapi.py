# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render retained declaration differences as actionable, evidence-linked findings."""

import argparse
import copy
import hashlib
import html
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from ..acquisition.http import write_json
from .findings import build_findings

# Default review order, not severity. Keep descriptions beside the presentation owner.
CATEGORY_GUIDE = (
    (
        "target_only",
        "Target-only declarations",
        "The target declares the field or response at the inspected location; Dynamo does not. "
        "Check aliases and possible gaps in request handling or response output.",
    ),
    (
        "both_declared_mismatch",
        "Differences in shared declarations",
        "Both sides declare the field or response, but their declarations differ. "
        "Check whether the difference affects accepted inputs or client-visible outputs.",
    ),
    (
        "representation_ambiguity",
        "Representation ambiguity",
        "Schema structure or limited comparison evidence prevents a clear conclusion. "
        "Resolve branches and comparison scope before inferring removal, restriction or equivalence.",
    ),
    (
        "missing_coverage",
        "Coverage gaps",
        "Missing declarations or tool limitations prevent assessment. "
        "Obtain the named evidence before drawing a compatibility conclusion.",
    ),
    (
        "dynamo_only",
        "Dynamo-only declarations",
        "Dynamo declares the field or response at the inspected location; the target does not. "
        "Check aliases and extension scope; extra output can still affect clients.",
    ),
    (
        "alias_evidence",
        "Input alias matches",
        "Declared aliases connect different input names. "
        "Review value constraints, parent requirements and duplicate-name handling separately.",
    ),
    (
        "intentional_exclusion",
        "Intentional exclusions",
        "The comparison deliberately excludes this scope. "
        "Confirm the exclusion remains intended; do not infer compatibility for excluded content.",
    ),
)


def prose(value) -> str:
    """Escape untrusted evidence for a single Markdown paragraph or table cell."""
    text = html.escape(str(value), quote=True).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", text)


def file_link(label: str, filename: str, location: str = "", prefix: str = "") -> str:
    """Link a retained relative file and optional JSON Pointer; reject external or escaping paths."""
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or ":" in filename or "\\" in filename:
        return prose(label) + " (unsafe evidence path omitted)"
    url = quote(prefix + filename, safe="/")
    if location:
        url += "#" + quote(location.removeprefix("#"), safe="/~")
    return f"[{prose(label)}]({url})"


def reported_change(row: dict) -> str:
    """Summarize literal member/scalar deltas without interpreting composition or null markers."""
    delta, change = row["delta"], row["change"]
    if not isinstance(delta, dict) or change in {
        "structure",
        "unknown",
        "missing_coverage",
        "alias_evidence",
        "intentional_exclusion",
    }:
        return ""
    label = {
        "required": "required-member",
        "listOfTypes": "type",
        "min": "minimum",
        "max": "maximum",
    }.get(change, change)
    parts = []
    for key in ("added", "deleted"):
        if isinstance(delta.get(key), list):
            values = [
                value if isinstance(value, str) else json.dumps(value)
                for value in delta[key]
            ]
            parts.append(key + " " + ", ".join(values))
    if "from" in delta or "to" in delta:
        parts.extend(
            key + " " + json.dumps(delta[key], sort_keys=True)
            for key in ("from", "to")
            if key in delta
        )
    if not parts:
        return ""
    result = "Reported " + label + " change: " + "; ".join(parts) + "."
    if any(delta.get(key, True) is None for key in ("from", "to")):
        result += " A null marker does not distinguish an absent annotation from a null value; check the declarations."
    return result


def render(
    report: dict, evidence_prefix: str = "", available_files: set | None = None
) -> str:
    """Render category descriptions and findings in review order, not severity order.

    Prefer report.finding_inventory from build_findings. Legacy callers may omit
    it; their declarations are explicitly unavailable. `evidence_prefix` names
    the retained evidence subdirectory for regenerated reports. When available_files
    is supplied, mark missing output-relative files instead of emitting broken links.
    Unknown categories follow the documented categories. Inputs are unchanged;
    rendering does not adjudicate runtime behavior or policy.
    """
    inventory = report.get("finding_inventory") or build_findings(report)

    def evidence_link(label, filename, location="", prefix=""):
        """Link only retained supporting evidence when an availability set was supplied."""
        if available_files is not None and prefix + filename not in available_files:
            return prose(label) + " (not retained)"
        return file_link(label, filename, location, prefix)

    counts, accounting = inventory["counts"], inventory["accounting"]
    lines = [
        f"# Dynamo versus {prose(report['framework_name'])} request/response-contract assessment",
        "",
        f"Status: **{prose(report['status'])}**. Behavioral conformance: **not assessed**.",
        "",
        "These are declaration candidates, not confirmed runtime defects. For requests, check "
        "reference-client inputs against Dynamo admission and effects. For responses, check "
        "possible Dynamo outputs against the client's expectations.",
        "",
        f"{counts['location_groups']} location groups; {counts['readable_findings']} readable findings. "
        f"All {accounting['raw_terminal_count']} raw candidate terminals are accounted for exactly once. "
        "These counts use different units; neither is a defect count or compatibility score.",
        "",
        "## Review actions",
        "",
        "1. Inspect both declarations and each suggested check below. Resolve structural ambiguity "
        "before calling a field removed or a value forbidden.",
        "2. Investigate coverage gaps. A missing default annotation is not a null default; "
        "alias names do not prove value compatibility.",
        "3. Attach revision-scoped evidence, an owner and a disposition to each candidate. "
        "Intentional policy and historical runtime observations require reviewer evidence. "
        "This generator supplies neither approval nor current runtime confirmation.",
        "",
        "## Categories in review order",
        "",
        "Use this order to start review, not to assign severity. Prioritize findings by "
        "impact and evidence. The target is the framework compared with Dynamo. "
        "Categories describe schema evidence, not runtime support.",
        "",
        "Exact declared alias matches appear as alias evidence, not one-sided field gaps. "
        "Alias names alone do not establish value compatibility. An absent default annotation "
        "does not make a field absent.",
        "",
        "| Category | Findings | Meaning and review focus |",
        "| --- | ---: | --- |",
    ]
    known = {name for name, _, _ in CATEGORY_GUIDE}
    categories = CATEGORY_GUIDE + tuple(
        (
            name,
            name,
            "Unrecognized category. Inspect the retained evidence before classifying it.",
        )
        for name in sorted(set(counts["categories"]) - known)
    )
    lines.extend(
        f"| {prose(name)} | {counts['categories'].get(name, 0)} | {description} |"
        for name, _, description in categories
    )
    lines.extend(
        [
            "",
            evidence_link("Complete inventory and accounting", "findings.json") + ". "
            "Location IDs remain stable when content changes. Compare delta and declaration hashes separately.",
        ]
    )
    for name, title, _ in categories:
        lines.extend(["", "## " + prose(title), ""])
        selected = [row for row in inventory["findings"] if row["category"] == name]
        if not selected:
            lines.append(
                "None reported in this category; this is not a runtime compatibility verdict."
            )
        for row in selected:
            scope = f"{row['endpoint']} · {row['transport']}"
            if row["kind"] in {"request", "response"}:
                scope += " · " + row["kind"]
            if row["status"]:
                scope += " · HTTP " + row["status"]
            lines.extend(
                [
                    "",
                    f"### {row['id']} — {prose(scope)} · {prose(row['field'])}",
                    "",
                    f"Category: {prose(row['category'])}. Direction: {prose(row['direction'])}.",
                    "",
                    "| Declaration | Recorded constraints |",
                    "| --- | --- |",
                    f"| Reference | {prose(row['declarations']['framework']['summary'])} |",
                    f"| Dynamo | {prose(row['declarations']['dynamo']['summary'])} |",
                    "",
                    prose(reported_change(row)),
                    "",
                    prose(row["interpretation"]),
                    "",
                    "**Next action:** " + prose(row["suggested_action"]),
                    "",
                    "Location: " + prose(row["location"]) + ".",
                ]
            )
            links = []
            for side, declaration in row["declarations"].items():
                for record in declaration["evidence"]:
                    links.append(
                        evidence_link(
                            side + " declaration " + record["pointer"],
                            record["file"],
                            record["pointer"],
                            evidence_prefix,
                        )
                    )
            links.extend(
                evidence_link("raw delta " + raw, "report.json", raw)
                for raw in row["raw_report_pointers"]
            )
            links.extend(
                evidence_link("alias evidence", "report.json", raw)
                for raw in row["alias_report_pointers"]
            )
            lines.extend(["", "Evidence: " + "; ".join(links) + "."])
            if row["candidate_ids"]:
                lines.append(
                    "Location groups: " + ", ".join(row["candidate_ids"]) + "."
                )
    lines.extend(
        [
            "",
            "## Supporting evidence",
            "",
            "Raw deltas remain unchanged: "
            + evidence_link(
                "request comparator output", "oasdiff.json", prefix=evidence_prefix
            )
            + "; "
            + evidence_link(
                "response comparator output",
                "response-oasdiff.json",
                prefix=evidence_prefix,
            )
            + ".",
            "",
            "### Equivalent representations",
            "",
            "Only supported equivalent encodings are normalized. Other union/allOf changes remain "
            "unresolved structural groups with complete nested deltas. See "
            + evidence_link(
                "request normalization", "normalization.json", prefix=evidence_prefix
            )
            + " and "
            + evidence_link(
                "response normalization",
                "response-normalization.json",
                prefix=evidence_prefix,
            )
            + ".",
            "",
            "### Source-qualified SSE interpretations",
            "",
            "Native itemSchema and explicitly selected event envelopes use encoding-tagged decoded data. "
            "Literal [DONE] remains distinct from JSON containing that string. See "
            + evidence_link(
                "source selections", "response-sources.json", prefix=evidence_prefix
            )
            + ".",
            "",
            "Chunk agreement is not event equivalence. Chunk-only pairs omit both SSE body schemas from "
            "the full response diff but retain status/media differences. Full projections remain in "
            "*.responses.json. Envelope, encoding, error, terminator and extension coverage remain separate. See "
            + evidence_link(
                "selected chunk probes",
                "response-chunk-probes.json",
                prefix=evidence_prefix,
            )
            + ".",
            "",
            "### Inputs and limits",
            "",
        ]
    )
    for side, metadata in sorted(report.get("inputs", {}).items()):
        lines.append(
            f"- {prose(side)} raw SHA-256: {prose(metadata['sha256'])}; "
            + evidence_link(
                "acquisition metadata",
                side + "/acquisition.json",
                prefix=evidence_prefix,
            )
            + "."
        )
    lines.extend(
        [
            "",
            "Caller source/revision annotations and legacy deployment metadata are unverified. "
            "Checksums and guarded composition verify input assumptions, not source identity or runtime behavior. "
            "The caller must select dependency-matching inputs.",
            "",
            "Inference, forwarding, runtime defaults, response headers, other endpoints and stream lifecycle "
            "are outside this assessment. Historical observations keep their original revisions and configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def load_projections(directory: Path) -> dict:
    """Read available normalized projections; missing files stay unavailable, invalid JSON fails."""
    filenames = (
        f"{side}.{kind}.normalized.json"
        for side in ("framework", "dynamo")
        for kind in ("requests", "responses")
    )
    return {
        name: json.loads((directory / name).read_text())
        for name in filenames
        if (directory / name).is_file()
    }


def write_reports(
    report: dict, directory: Path, projections: dict, evidence_prefix: str = ""
) -> dict:
    """Write additive JSON, inventory and Markdown into an existing output directory.

    Preserve report keys except regenerated finding_inventory. Inputs are unchanged.
    I/O or accounting failures propagate and may leave partial files.
    """
    result = copy.deepcopy(report)
    result["finding_inventory"] = build_findings(result, projections)
    for name, value in (
        ("report.json", result),
        ("findings.json", result["finding_inventory"]),
    ):
        write_json(directory / name, value)
    available = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    (directory / "report.md").write_text(render(result, evidence_prefix, available))
    return result


def regenerate(assessment_dir: Path, output_dir: Path) -> dict:
    """Re-render saved evidence without comparison into a new portable output directory.

    Reject an existing output, an output inside the input, or input symlinks.
    Copy original evidence under evidence/ so links survive relocation. Missing
    projections become explicit limitations. Failures may leave partial output;
    retry with a new directory. The source assessment is never modified.
    """
    source, target = assessment_dir.resolve(), output_dir.resolve()
    if target == source or source in target.parents:
        raise ValueError("Output directory must not be inside the frozen assessment")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Frozen assessment must not contain symlinks")
    report_bytes = (source / "report.json").read_bytes()
    report = json.loads(report_bytes)
    projections = load_projections(source)
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, target / "evidence")
    result = write_reports(report, target, projections, "evidence/")
    write_json(
        target / "render-source.json",
        {
            "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "comparison_rerun": False,
            "projection_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in sorted(projections)
            },
        },
    )
    return result


def main(argv=None) -> int:
    """Render a saved assessment; exit 2 on input, I/O or accounting failure, otherwise return 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = regenerate(args.assessment_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Report rendering failed: {error}\n")
    print(
        f"Rendered {report['finding_inventory']['counts']['readable_findings']} findings: {args.output_dir / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

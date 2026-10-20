# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write assessment reports and rebuild them from saved comparison evidence.

findings.py owns interpretation; this module owns the JSON/Markdown files,
category descriptions and review order, and escaped links to retained evidence.
write_reports serves new assessments. regenerate copies an existing assessment
into evidence/ and rebuilds its findings and reports with the current interpreter.
Interpretations can change; raw deltas do not, and comparison is not rerun.

report.json carries the assessment and finding inventory; report.md presents it
for human review. See README.md for CLI usage and the distinction between findings
and approved changes.
"""

import argparse
import copy
import hashlib
import html
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from ..inputs import write_json
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
    """Format an evidence link; unsafe filenames become explanatory text, not links.

    `prefix` is a trusted relative directory supplied by the renderer, not evidence
    content. `location` is an optional JSON Pointer. File existence is not checked.
    """
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

    Use report.finding_inventory when present; otherwise build it with declarations
    marked unavailable. `evidence_prefix` is the trusted relative evidence directory,
    including its trailing slash, or empty for a new assessment. When available_files
    is supplied, mark missing output-relative files instead of emitting broken links.
    Put target-only request/response guidance below its category heading.
    Omit empty sections and generic per-finding guidance; retain specific checks
    and limitations. Unknown categories follow the documented categories. Inputs
    are unchanged; rendering does not adjudicate runtime behavior or policy.
    """
    inventory = report.get("finding_inventory") or build_findings(report)

    def evidence_link(label, filename, location="", prefix=""):
        """Link only retained supporting evidence when an availability set was supplied."""
        if available_files is not None and prefix + filename not in available_files:
            return prose(label) + " (not retained)"
        return file_link(label, filename, location, prefix)

    counts = inventory["counts"]
    lines = [
        f"# Dynamo versus {prose(report['framework_name'])} request/response-contract assessment",
        "",
        f"Status: **{prose(report['status'])}**. Behavioral conformance: **not assessed**.",
        "",
        "These are schema findings, not confirmed runtime defects or approved fixes.",
        "",
        f"{counts['readable_findings']} findings.",
        "",
        "## Review actions",
        "",
        "1. Inspect both declarations and linked evidence. Resolve structural ambiguity and coverage gaps.",
        "2. In pinned code, check target inputs against Dynamo validation and backend effects. "
        "For responses, check Dynamo output against target-client decoding, including JSON/SSE errors.",
        "3. Record revision-scoped evidence and a disposition before proposing changes.",
        "",
        "## Categories in review order",
        "",
        "Use this order to start review, not to assign severity. Prioritize findings by "
        "impact and evidence. The target is the framework compared with Dynamo.",
        "",
        "Exact declared alias matches appear as alias evidence, not one-sided field gaps.",
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
            evidence_link("Structured assessment", "report.json") + ". "
            "Finding IDs identify locations and change kinds, not unchanged content.",
        ]
    )
    for name, title, _ in categories:
        selected = [
            (index, row)
            for index, row in enumerate(inventory["findings"])
            if row["category"] == name
        ]
        if not selected:
            continue
        lines.extend(["", "## " + prose(title), ""])
        if name == "target_only":
            lines.extend(
                [
                    "**Requests:** Reference-only declaration at this location. "
                    "Extra accepted input is not itself a client incompatibility; "
                    "missing input declarations do not prove rejection.",
                    "",
                    "**Next action:** Check the missing field against pinned validation/translation "
                    "code, then probe its declared boundary and backend effect.",
                    "",
                    "**Responses:** Reference-only declaration at this location. "
                    "Extra possible output can affect clients; missing declarations do not "
                    "prove the output never occurs.",
                    "",
                    "**Next action:** Check the missing field or response against the serializer "
                    "and client decoder; include JSON/SSE error cases before changing the response contract.",
                    "",
                ]
            )
        for index, row in selected:
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
                    "| Declaration | Recorded constraints |",
                    "| --- | --- |",
                    f"| Reference | {prose(row['declarations']['framework']['summary'])} |",
                    f"| Dynamo | {prose(row['declarations']['dynamo']['summary'])} |",
                    "",
                ]
            )
            change = reported_change(row)
            if change:
                lines.extend([prose(change), ""])
            if (
                row["change"] in {"structure", "unknown", "default", "required"}
                or row["category"]
                not in {"target_only", "dynamo_only", "both_declared_mismatch"}
                or row["alias_report_pointers"]
            ):
                lines.extend(
                    [
                        prose(row["interpretation"]),
                        "",
                        "**Next check:** " + prose(row["suggested_action"]),
                        "",
                    ]
                )
            links = [
                evidence_link(
                    "Finding details",
                    "report.json",
                    f"#/finding_inventory/findings/{index}",
                )
            ]
            for side, declaration in row["declarations"].items():
                for number, record in enumerate(declaration["evidence"], 1):
                    links.append(
                        evidence_link(
                            f"{'Target' if side == 'framework' else 'Dynamo'} declaration {number}",
                            record["file"],
                            record["pointer"],
                            evidence_prefix,
                        )
                    )
            links.extend(
                evidence_link(f"Raw delta {number}", "report.json", raw)
                for number, raw in enumerate(row["raw_report_pointers"], 1)
            )
            links.extend(
                evidence_link("alias evidence", "report.json", raw)
                for raw in row["alias_report_pointers"]
            )
            lines.extend(["", "Evidence: " + "; ".join(links) + "."])
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
                "input metadata",
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
    """Map retained request/response projection filenames to parsed JSON.

    Read the four framework/Dynamo normalized projections under directory.
    Omit missing files; JSON and I/O errors propagate for files that exist.
    """
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
    """Rebuild findings and overwrite report.json and report.md.

    directory must exist; projections follows build_findings' filename-to-document
    mapping. evidence_prefix follows render's relative-directory convention.
    Return a copy of report with finding_inventory replaced; other keys and caller
    inputs are unchanged. Raw evidence files are not written by this function.
    I/O or accounting failures propagate and may leave partial files.
    """
    result = copy.deepcopy(report)
    result["finding_inventory"] = build_findings(result, projections)
    write_json(directory / "report.json", result)
    available = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    (directory / "report.md").write_text(render(result, evidence_prefix, available))
    return result


def regenerate(assessment_dir: Path, output_dir: Path) -> dict:
    """Rebuild a saved assessment's findings and reports without rerunning comparison.

    assessment_dir must contain report.json; normalized projections are optional.
    Reject an existing output, an output inside the input, or input symlinks.
    Copy original evidence under evidence/ so links survive relocation, and record
    source hashes in render-source.json. Return the rebuilt assessment; current
    interpretation code can change categories or suggestions, but not raw deltas.
    Missing projections become explicit limitations. Failures may leave partial
    output; retry with a new directory. The source assessment is never modified.
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

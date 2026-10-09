# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A compact reading order around the complete library-produced request diff."""

import json


def render(report: dict) -> str:
    framework = report["framework_name"]
    lines = [
        f"# Dynamo versus {framework} request-contract assessment",
        "",
        f"Status: **{report['status']}**. Behavioral conformance: **not assessed**.",
        "",
        f"Direction: {framework} → Dynamo. Added means declared by Dynamo only; "
        f"deleted means declared by {framework} only. These are schema differences, not "
        "automatically confirmed runtime incompatibilities.",
        "",
        "## Inputs",
        "",
    ]
    for side, metadata in report["inputs"].items():
        lines.append(
            f"- {side}: raw SHA-256 `{metadata['sha256']}` "
            f"([input metadata]({side}/acquisition.json))."
        )
    lines.extend(
        [
            "",
            "Exact input documents and comparison settings are retained. "
            "Caller annotations and legacy deployment metadata are unverified; "
            "server identity, source, image and configuration are not attested. "
            "Deployment provenance is not required for declared-contract comparison.",
            "",
            "The composition manifest declares its expected Dynamo dependencies. "
            "Checking that the selected document corresponds to those dependencies "
            "is the caller's responsibility; checksum and schema-patch guards still apply.",
            "",
            "## Coverage gaps — address before claiming complete coverage",
            "",
        ]
    )
    for gap in report["coverage_gaps"]:
        lines.append(
            f"- **{gap['side']} {gap['endpoint']}** `{gap.get('field', '*')}`: "
            f"{gap['reason']} Location: `{gap['location']}`."
        )
    if not report["coverage_gaps"]:
        lines.append(
            "No known gaps were detected by the configured coverage checks. "
            "This is not a proof of equivalent accepted requests."
        )
    lines.extend(
        [
            "",
            "## Equivalent representations",
            "",
            "Comparison distinguishes equivalent representations from meaningful "
            "contract differences. Bare string-or-null `anyOf` and string/null type "
            "arrays are compared as equivalent encodings; all sibling constraints "
            "and annotations remain. This is not a claim that entire fields or "
            "endpoints are equivalent.",
            "",
            "Original request projections are retained as `*.requests.json`; "
            "oasdiff compares `*.requests.normalized.json`. "
            "See [normalization.json](normalization.json) for each rewrite.",
        ]
    )
    for change in report.get("representation_normalizations", []):
        lines.append(
            f"- `{change['side']}` `{change['location']}`: `{change['rule']}`."
        )
    lines.extend(
        [
            "",
            "## Input alias matches",
            "",
            "Explicit exporter metadata matches backend spellings to Dynamo fields at "
            "the same request instance path. Name coverage is not compatibility. "
            "Value probes compare constraints/defaults; parent requiredness, union "
            "conditions and simultaneous-name handling remain separate. "
            "The complete request diff is retained unchanged. "
            "See [alias-matches.json](alias-matches.json) for pointers and parent context.",
        ]
    )
    for match in report.get("alias_matches", []):
        lines.append(
            f"- `{match['endpoint']}` `{'.'.join('[]' if part is None else part for part in match['path'])}` → "
            f"`{match['dynamo_name']}`: **name covered via alias**; "
            f"value schema: `{match['value_comparison']}` "
            f"([diff]({match['evidence']})). {match['simultaneous_names']}."
        )
    if not report.get("alias_matches"):
        lines.append(
            "No unambiguous declared alias matches found; no aliases inferred."
        )
    lines.extend(["", "## Intentional exclusions", ""])
    for exclusion in report["exclusions"]:
        lines.append(f"- `{exclusion['field']}`: {exclusion['reason']}")
    lines.extend(
        [
            "",
            "Responses, inference behavior, forwarding, runtime defaults and "
            "streaming behavior are outside this request-schema assessment.",
            "",
            "## Declared request differences",
            "",
            "See [report.json](report.json) for all structured findings and "
            "[oasdiff.json](oasdiff.json) for the unabridged library result on "
            "the normalized comparison inputs. "
            "Structural union/nullability differences can be representation-only; "
            "review their JSON before calling them wire incompatibilities.",
            "",
        ]
    )
    for finding in report["differences"]:
        lines.extend(
            [
                f"### POST {finding['endpoint']}",
                "",
                f"Input schema location (both retained specs): `{finding['location']}`.",
                "",
            ]
        )
        delta = finding["delta"]
        schema = (
            delta.get("content", {})
            .get("modified", {})
            .get("application/json", {})
            .get("schema", {})
        )
        properties = schema.get("properties", {})
        top_level_aliases = [
            match
            for match in report.get("alias_matches", [])
            if match["endpoint"] == finding["endpoint"] and len(match["path"]) == 1
        ]
        for key, title in (
            ("deleted", f"{framework}-only declarations"),
            ("added", "Dynamo-only declarations"),
        ):
            covered = {
                match["backend_name" if key == "deleted" else "dynamo_name"]
                for match in top_level_aliases
            }
            names = set(properties.get(key, [])) - covered
            lines.append(
                f"- {title}: "
                + (", ".join(f"`{name}`" for name in sorted(names)) or "none")
            )
        for match in top_level_aliases:
            lines.append(
                f"- Alias-related spelling difference: `{match['backend_name']}` "
                f"is accepted as `{match['dynamo_name']}`. Value/parent differences "
                "are not suppressed; see Input alias matches and complete delta."
            )
        changed = properties.get("modified", {})
        lines.append(
            "- Changed fields: "
            + (", ".join(f"`{name}`" for name in sorted(changed)) or "none")
        )
        if schema.get("required"):
            lines.append(
                f"- Required-field changes: `{json.dumps(schema['required'], sort_keys=True)}`"
            )
        lines.extend(
            [
                "",
                "<details>",
                "<summary>Complete request-body delta</summary>",
                "",
                "```json",
                json.dumps(delta, indent=2, sort_keys=True),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    if not report["differences"]:
        lines.append(
            "No declared request-body differences reported. Coverage gaps still apply."
        )
    lines.extend(
        [
            "",
            "## Next actions",
            "",
            "1. Resolve or explicitly track coverage gaps; do not approve them as compatibility.",
            "2. Review each declared difference as an intended divergence, a schema-export defect, "
            "or a protocol change needing implementation.",
            "3. Reacquire and reassess after fixes or dependency/framework version bumps.",
            "4. Run behavioral conformance separately; this report does not satisfy that gate.",
            "",
        ]
    )
    return "\n".join(lines)

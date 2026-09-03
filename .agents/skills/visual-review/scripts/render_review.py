#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render and validate a self-contained visual code-review dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
DIFF_METADATA_PREFIXES = (
    "index ",
    "--- ",
    "+++ ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)
DIFF_BINARY_PREFIXES = ("Binary files ", "GIT binary patch")
VALID_SEVERITIES = {"P0", "P1", "P2", "P3", "note"}
VALID_SIDES = {"old", "new"}
VALID_MARKER_KINDS = {"relevant", "risk", "test", "out"}
VALID_TEST_TONES = {"required", "existing", "neutral"}
VALID_HEAT_LEVELS = {"high", "medium", "low"}
VALID_SMELL_LEVELS = {"high", "medium", "none"}
VALID_DIAGRAM_TYPES = {"component", "sequence"}
VALID_DIAGRAM_TONES = {"danger", "warning", "success", "info", "neutral"}
VALID_COMPONENT_KINDS = {"changed", "existing", "external", "test"}


class ReviewError(ValueError):
    pass


def slug(value: str) -> str:
    return re.sub(r"^-|-$", "", re.sub(r"[^A-Za-z0-9]+", "-", value))


def require_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ReviewError(f"{field} must be a non-empty string")
    return value


def require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReviewError(f"{field} must be an array")
    return value


def require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ReviewError(f"{field} must be a positive integer")
    return value


def github_pr_files_url(source_url: str) -> str:
    """Normalize a GitHub pull-request URL to its current files view."""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "github.com",
        "www.github.com",
    }:
        return ""
    match = re.fullmatch(
        r"/([^/]+)/([^/]+)/pull/(\d+)(?:/(?:files|changes))?/?", parsed.path
    )
    if not match:
        return ""
    files_path = f"/{match.group(1)}/{match.group(2)}/pull/{match.group(3)}/files"
    return urlunsplit((parsed.scheme, parsed.netloc, files_path, "", ""))


def github_diff_anchors(source_url: str, files: list[dict[str, Any]]) -> dict[str, str]:
    """Return GitHub PR file anchors keyed by the post-change path."""
    files_url = github_pr_files_url(source_url)
    if not files_url:
        return {}
    return {
        item[
            "path"
        ]: f"{files_url}#diff-{hashlib.sha256(item['path'].encode('utf-8')).hexdigest()}"
        for item in files
    }


def gitlab_mr_diffs_url(source_url: str) -> str:
    """Normalize a GitLab merge-request URL to its current diffs view."""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    match = re.fullmatch(
        r"/(.+)/-/merge_requests/(\d+)(?:/(?:diffs|changes))?/?", parsed.path
    )
    if not match:
        return ""
    diffs_path = f"/{match.group(1)}/-/merge_requests/{match.group(2)}/diffs"
    return urlunsplit((parsed.scheme, parsed.netloc, diffs_path, "", ""))


def gitlab_file_hash(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def gitlab_diff_anchors(source_url: str, files: list[dict[str, Any]]) -> dict[str, str]:
    """Return pinned GitLab merge-request file anchors keyed by changed path."""
    diffs_url = gitlab_mr_diffs_url(source_url)
    if not diffs_url:
        return {}
    anchors = {}
    for item in files:
        file_hash = gitlab_file_hash(item["path"])
        anchors[item["path"]] = f"{diffs_url}?pin={file_hash}#{file_hash}"
    return anchors


def gitlab_line_anchors(
    source_url: str, diff: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Return exact GitLab merge-request line anchors grouped by changed path."""
    diffs_url = gitlab_mr_diffs_url(source_url)
    if not diffs_url:
        return {}
    anchors: dict[str, dict[str, str]] = {}
    for (path, side, line), (old_position, new_position) in diff[
        "line_positions"
    ].items():
        file_hash = gitlab_file_hash(path)
        line_code = f"{file_hash}_{old_position}_{new_position}"
        anchors.setdefault(path, {})[
            f"{side}:{line}"
        ] = f"{diffs_url}?pin={file_hash}#{line_code}"
    return anchors


def parse_diff(raw: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    targets: set[str] = set()
    line_keys: set[tuple[str, str, int]] = set()
    line_positions: dict[tuple[str, str, int], tuple[int, int]] = {}
    current: dict[str, Any] | None = None
    old_line = new_line = 0
    old_remaining = new_remaining = 0
    skip_binary_payload = False

    for number, line in enumerate(
        raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1
    ):
        if line.startswith("diff --git "):
            if old_remaining or new_remaining:
                raise ReviewError(
                    f"diff line {number}: previous hunk ended before its declared line counts"
                )
            match = DIFF_RE.match(line)
            if not match:
                raise ReviewError(f"diff line {number}: malformed file header")
            current = {
                "path": match.group(2),
                "additions": 0,
                "deletions": 0,
                "rows": [],
            }
            files.append(current)
            targets.add(f"file-{slug(current['path'])}")
            old_remaining = new_remaining = 0
            skip_binary_payload = False
            continue

        if current is None:
            if line.strip():
                raise ReviewError(
                    f"diff line {number}: content before first file header"
                )
            continue

        if skip_binary_payload:
            continue

        if line.startswith("@@"):
            if old_remaining or new_remaining:
                raise ReviewError(
                    f"diff line {number}: previous hunk ended before its declared line counts"
                )
            match = HUNK_RE.match(line)
            if not match:
                raise ReviewError(f"diff line {number}: malformed hunk header")
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            old_remaining = int(match.group(2) or 1)
            new_remaining = int(match.group(4) or 1)
            current["rows"].append({"type": "hunk", "text": line})
            continue

        in_hunk = old_remaining > 0 or new_remaining > 0
        if not in_hunk and line.startswith(DIFF_BINARY_PREFIXES):
            current["rows"].append({"type": "meta", "text": line})
            skip_binary_payload = True
            continue

        if not in_hunk and line.startswith(DIFF_METADATA_PREFIXES):
            current["rows"].append({"type": "meta", "text": line})
            continue

        if line.startswith("\\"):
            current["rows"].append({"type": "meta", "text": line})
            continue

        if not in_hunk:
            if not line:
                continue
            raise ReviewError(
                f"diff line {number}: content row outside a unified-diff hunk {line[:40]!r}"
            )

        file_slug = slug(current["path"])
        if line.startswith("+"):
            if new_remaining < 1:
                raise ReviewError(
                    f"diff line {number}: addition exceeds the hunk's new-line count"
                )
            current["additions"] += 1
            key = (current["path"], "new", new_line)
            line_keys.add(key)
            line_positions[key] = (old_line, new_line)
            targets.add(f"line-{file_slug}-new-{new_line}")
            current["rows"].append(
                {
                    "type": "add",
                    "text": line[1:],
                    "oldLine": None,
                    "newLine": new_line,
                    "oldAnchor": old_line,
                    "newAnchor": new_line,
                }
            )
            new_line += 1
            new_remaining -= 1
        elif line.startswith("-"):
            if old_remaining < 1:
                raise ReviewError(
                    f"diff line {number}: deletion exceeds the hunk's old-line count"
                )
            current["deletions"] += 1
            key = (current["path"], "old", old_line)
            line_keys.add(key)
            line_positions[key] = (old_line, new_line)
            targets.add(f"line-{file_slug}-old-{old_line}")
            current["rows"].append(
                {
                    "type": "del",
                    "text": line[1:],
                    "oldLine": old_line,
                    "newLine": None,
                    "oldAnchor": old_line,
                    "newAnchor": new_line,
                }
            )
            old_line += 1
            old_remaining -= 1
        elif line.startswith(" "):
            if old_remaining < 1 or new_remaining < 1:
                raise ReviewError(
                    f"diff line {number}: context exceeds the hunk's declared line counts"
                )
            old_key = (current["path"], "old", old_line)
            new_key = (current["path"], "new", new_line)
            line_keys.add(old_key)
            line_keys.add(new_key)
            line_positions[old_key] = (old_line, new_line)
            line_positions[new_key] = (old_line, new_line)
            targets.add(f"line-{file_slug}-old-{old_line}")
            targets.add(f"line-{file_slug}-new-{new_line}")
            current["rows"].append(
                {
                    "type": "context",
                    "text": line[1:],
                    "oldLine": old_line,
                    "newLine": new_line,
                    "oldAnchor": old_line,
                    "newAnchor": new_line,
                }
            )
            old_line += 1
            new_line += 1
            old_remaining -= 1
            new_remaining -= 1
        else:
            raise ReviewError(
                f"diff line {number}: unsupported unified-diff row {line[:40]!r}"
            )

    if old_remaining or new_remaining:
        raise ReviewError("diff ended before its final hunk's declared line counts")
    if not files:
        raise ReviewError("diff contains no files")

    return {
        "files": files,
        "changed_files": len(files),
        "additions": sum(item["additions"] for item in files),
        "deletions": sum(item["deletions"] for item in files),
        "targets": targets,
        "line_keys": line_keys,
        "line_positions": line_positions,
    }


def register_id(ids: set[str], item_id: Any, field: str) -> str:
    value = require_string(item_id, field)
    if value in ids:
        raise ReviewError(f"duplicate id {value!r}")
    ids.add(value)
    return value


def validate_links(links: Any, field: str, targets: set[str]) -> None:
    for index, link in enumerate(require_list(links, field)):
        if not isinstance(link, dict):
            raise ReviewError(f"{field}[{index}] must be an object")
        require_string(link.get("label"), f"{field}[{index}].label")
        target = require_string(link.get("target"), f"{field}[{index}].target")
        if target not in targets:
            raise ReviewError(
                f"{field}[{index}].target references unknown id {target!r}"
            )


def validate_finding_ids(
    value: Any, field: str, finding_ids: set[str], *, allow_empty: bool = False
) -> list[str]:
    related = require_list(value, field)
    if not related and not allow_empty:
        raise ReviewError(f"{field} must not be empty")
    if len(related) > 1:
        raise ReviewError(f"{field} must attach the artifact to exactly one finding")
    seen: set[str] = set()
    for index, item in enumerate(related):
        finding_id = require_string(item, f"{field}[{index}]")
        if finding_id not in finding_ids:
            raise ReviewError(
                f"{field}[{index}] references unknown finding {finding_id!r}"
            )
        if finding_id in seen:
            raise ReviewError(f"{field} contains duplicate finding {finding_id!r}")
        seen.add(finding_id)
    return related


def validate_score(score: Any, field: str) -> None:
    if not isinstance(score, dict):
        raise ReviewError(f"{field} must be an object")
    base = score.get("base")
    value = score.get("value")
    if not isinstance(base, int) or isinstance(base, bool) or not 0 <= base <= 10:
        raise ReviewError(f"{field}.base must be an integer from 0 through 10")
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10:
        raise ReviewError(f"{field}.value must be an integer from 1 through 10")
    require_string(score.get("summary"), f"{field}.summary")
    total = base
    for index, factor in enumerate(
        require_list(score.get("factors"), f"{field}.factors")
    ):
        prefix = f"{field}.factors[{index}]"
        if not isinstance(factor, dict):
            raise ReviewError(f"{prefix} must be an object")
        require_string(factor.get("label"), f"{prefix}.label")
        delta = factor.get("delta")
        if (
            not isinstance(delta, int)
            or isinstance(delta, bool)
            or not -10 <= delta <= 10
        ):
            raise ReviewError(f"{prefix}.delta must be an integer from -10 through 10")
        total += delta
    expected = max(1, min(10, total))
    if value != expected:
        raise ReviewError(
            f"{field}.value must equal clamp(base + sum(delta), 1, 10): expected {expected}, got {value}"
        )


def validate_spec(spec: dict[str, Any], diff: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ReviewError("review spec must be a JSON object")
    if spec.get("version", 1) != 1:
        raise ReviewError("only review spec version 1 is supported")
    require_string(spec.get("title"), "title")
    validate_score(spec.get("correctness_score"), "correctness_score")
    validate_score(spec.get("risk_score"), "risk_score")

    findings = require_list(spec.get("findings"), "findings")
    flows = require_list(spec.get("flows", []), "flows")
    file_map = require_list(spec.get("file_map", []), "file_map")
    matrix = require_list(spec.get("test_matrix", []), "test_matrix")
    references = require_list(spec.get("references", []), "references")
    code_blocks = require_list(spec.get("code_blocks", []), "code_blocks")
    diagrams = require_list(spec.get("diagrams", []), "diagrams")
    manifests = require_list(spec.get("manifests", []), "manifests")
    collapsed_files = require_list(
        spec.get("github_collapsed_files", []), "github_collapsed_files"
    )

    changed_paths = {item["path"] for item in diff["files"]}
    seen_collapsed: set[str] = set()
    for index, file_path in enumerate(collapsed_files):
        path = require_string(file_path, f"github_collapsed_files[{index}]")
        if path not in changed_paths:
            raise ReviewError(
                f"github_collapsed_files[{index}] references unchanged path {path!r}"
            )
        if path in seen_collapsed:
            raise ReviewError(
                f"github_collapsed_files contains duplicate path {path!r}"
            )
        seen_collapsed.add(path)

    ids = {
        "review-summary",
        "findings-overview",
        "scope-visual-review",
        "test-matrix",
        "logical-flows",
        "diagrams",
        "manifests",
        "file-map",
        "references",
        "diff-root",
    }
    ids.update(diff["targets"])

    for index, item in enumerate(findings):
        if not isinstance(item, dict):
            raise ReviewError(f"findings[{index}] must be an object")
        register_id(ids, item.get("id"), f"findings[{index}].id")
    for index, item in enumerate(flows):
        if not isinstance(item, dict):
            raise ReviewError(f"flows[{index}] must be an object")
        register_id(ids, item.get("id"), f"flows[{index}].id")
    for index, item in enumerate(references):
        if not isinstance(item, dict):
            raise ReviewError(f"references[{index}] must be an object")
        register_id(ids, item.get("id"), f"references[{index}].id")
    for index, item in enumerate(code_blocks):
        if not isinstance(item, dict):
            raise ReviewError(f"code_blocks[{index}] must be an object")
        register_id(ids, item.get("id"), f"code_blocks[{index}].id")
    for index, item in enumerate(matrix):
        if not isinstance(item, dict):
            raise ReviewError(f"test_matrix[{index}] must be an object")
        if item.get("id") is not None:
            register_id(ids, item.get("id"), f"test_matrix[{index}].id")
    for index, item in enumerate(diagrams):
        if not isinstance(item, dict):
            raise ReviewError(f"diagrams[{index}] must be an object")
        register_id(ids, item.get("id"), f"diagrams[{index}].id")
    for index, item in enumerate(manifests):
        if not isinstance(item, dict):
            raise ReviewError(f"manifests[{index}] must be an object")
        register_id(ids, item.get("id"), f"manifests[{index}].id")

    finding_ids = {finding["id"] for finding in findings}

    for index, finding in enumerate(findings):
        prefix = f"findings[{index}]"
        severity = require_string(finding.get("severity"), f"{prefix}.severity")
        if severity not in VALID_SEVERITIES:
            raise ReviewError(
                f"{prefix}.severity must be one of {sorted(VALID_SEVERITIES)}"
            )
        require_string(finding.get("title"), f"{prefix}.title")
        require_string(finding.get("summary"), f"{prefix}.summary")
        details = require_list(finding.get("details"), f"{prefix}.details")
        if len(details) < 2:
            raise ReviewError(
                f"{prefix}.details must contain at least two labeled groups"
            )
        for detail_index, detail in enumerate(details):
            detail_prefix = f"{prefix}.details[{detail_index}]"
            if not isinstance(detail, dict):
                raise ReviewError(f"{detail_prefix} must be an object")
            require_string(detail.get("label"), f"{detail_prefix}.label")
            items = require_list(detail.get("items"), f"{detail_prefix}.items")
            if not items:
                raise ReviewError(f"{detail_prefix}.items must not be empty")
            for item_index, item in enumerate(items):
                require_string(item, f"{detail_prefix}.items[{item_index}]")
        suggested_fix = finding.get("suggested_fix")
        if not isinstance(suggested_fix, dict):
            raise ReviewError(f"{prefix}.suggested_fix must be an object")
        require_string(suggested_fix.get("summary"), f"{prefix}.suggested_fix.summary")
        steps = require_list(
            suggested_fix.get("steps"), f"{prefix}.suggested_fix.steps"
        )
        if not steps:
            raise ReviewError(f"{prefix}.suggested_fix.steps must not be empty")
        for step_index, step in enumerate(steps):
            require_string(step, f"{prefix}.suggested_fix.steps[{step_index}]")
        for test_index, test in enumerate(
            require_list(
                suggested_fix.get("tests", []), f"{prefix}.suggested_fix.tests"
            )
        ):
            require_string(test, f"{prefix}.suggested_fix.tests[{test_index}]")
        require_string(finding.get("agent_prompt"), f"{prefix}.agent_prompt")
        file_path = require_string(finding.get("file"), f"{prefix}.file")
        side = require_string(finding.get("side"), f"{prefix}.side")
        if side not in VALID_SIDES:
            raise ReviewError(f"{prefix}.side must be old or new")
        line = require_positive_int(finding.get("line"), f"{prefix}.line")
        if (file_path, side, line) not in diff["line_keys"]:
            raise ReviewError(
                f"{prefix} points outside the diff: {file_path}:{side}:{line}"
            )
        involves_api_objects = finding.get("involves_api_objects", False)
        if not isinstance(involves_api_objects, bool):
            raise ReviewError(f"{prefix}.involves_api_objects must be a boolean")
        validate_links(finding.get("links", []), f"{prefix}.links", ids)

    for index, flow in enumerate(flows):
        prefix = f"flows[{index}]"
        validate_finding_ids(
            flow.get("finding_ids"), f"{prefix}.finding_ids", finding_ids
        )
        require_string(flow.get("title"), f"{prefix}.title")
        require_string(
            flow.get("description", ""), f"{prefix}.description", allow_empty=True
        )
        for step_index, step in enumerate(
            require_list(flow.get("steps", []), f"{prefix}.steps")
        ):
            if not isinstance(step, dict):
                raise ReviewError(f"{prefix}.steps[{step_index}] must be an object")
            require_string(step.get("title"), f"{prefix}.steps[{step_index}].title")
            target = require_string(
                step.get("target"), f"{prefix}.steps[{step_index}].target"
            )
            if target not in ids:
                raise ReviewError(
                    f"{prefix}.steps[{step_index}].target references unknown id {target!r}"
                )

    primary_diagrams = [
        diagram for diagram in diagrams if diagram.get("primary") is True
    ]
    if len(primary_diagrams) != 1 or primary_diagrams[0].get("type") != "component":
        raise ReviewError("diagrams must contain exactly one primary component diagram")

    for index, diagram in enumerate(diagrams):
        prefix = f"diagrams[{index}]"
        primary = diagram.get("primary", False)
        if not isinstance(primary, bool):
            raise ReviewError(f"{prefix}.primary must be a boolean")
        related_findings = validate_finding_ids(
            diagram.get("finding_ids", []),
            f"{prefix}.finding_ids",
            finding_ids,
            allow_empty=primary,
        )
        if primary and related_findings:
            raise ReviewError(
                f"{prefix}.finding_ids must be empty for the PR-wide primary diagram"
            )
        diagram_type = require_string(diagram.get("type"), f"{prefix}.type")
        if diagram_type not in VALID_DIAGRAM_TYPES:
            raise ReviewError(
                f"{prefix}.type must be one of {sorted(VALID_DIAGRAM_TYPES)}"
            )
        require_string(diagram.get("title"), f"{prefix}.title")
        require_string(
            diagram.get("description", ""), f"{prefix}.description", allow_empty=True
        )

        if diagram_type == "component":
            nodes = require_list(diagram.get("nodes"), f"{prefix}.nodes")
            if not nodes:
                raise ReviewError(f"{prefix}.nodes must not be empty")
            node_ids: set[str] = set()
            for node_index, node in enumerate(nodes):
                node_prefix = f"{prefix}.nodes[{node_index}]"
                if not isinstance(node, dict):
                    raise ReviewError(f"{node_prefix} must be an object")
                node_id = require_string(node.get("id"), f"{node_prefix}.id")
                if node_id in node_ids:
                    raise ReviewError(
                        f"{prefix}.nodes contains duplicate id {node_id!r}"
                    )
                node_ids.add(node_id)
                require_string(node.get("label"), f"{node_prefix}.label")
                require_string(
                    node.get("detail", ""), f"{node_prefix}.detail", allow_empty=True
                )
                kind = node.get("kind", "existing")
                if kind not in VALID_COMPONENT_KINDS:
                    raise ReviewError(
                        f"{node_prefix}.kind must be one of {sorted(VALID_COMPONENT_KINDS)}"
                    )
                for axis in ("x", "y"):
                    value = node.get(axis)
                    if value is not None and (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not 0 <= value <= 100
                    ):
                        raise ReviewError(
                            f"{node_prefix}.{axis} must be a number from 0 through 100 when set"
                        )
                target = node.get("target")
                if target is not None and target not in ids:
                    raise ReviewError(
                        f"{node_prefix}.target references unknown id {target!r}"
                    )
            for edge_index, edge in enumerate(
                require_list(diagram.get("edges"), f"{prefix}.edges")
            ):
                edge_prefix = f"{prefix}.edges[{edge_index}]"
                if not isinstance(edge, dict):
                    raise ReviewError(f"{edge_prefix} must be an object")
                for endpoint in ("from", "to"):
                    value = require_string(
                        edge.get(endpoint), f"{edge_prefix}.{endpoint}"
                    )
                    if value not in node_ids:
                        raise ReviewError(
                            f"{edge_prefix}.{endpoint} references unknown node {value!r}"
                        )
                require_string(edge.get("label"), f"{edge_prefix}.label")
                tone = edge.get("tone", "neutral")
                if tone not in VALID_DIAGRAM_TONES:
                    raise ReviewError(
                        f"{edge_prefix}.tone must be one of {sorted(VALID_DIAGRAM_TONES)}"
                    )
                for axis in ("label_x", "label_y"):
                    value = edge.get(axis)
                    if value is not None and (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not 0 <= value <= 100
                    ):
                        raise ReviewError(
                            f"{edge_prefix}.{axis} must be a number from 0 through 100"
                        )
                target = edge.get("target")
                if target is not None and target not in ids:
                    raise ReviewError(
                        f"{edge_prefix}.target references unknown id {target!r}"
                    )
        else:
            participants = require_list(
                diagram.get("participants"), f"{prefix}.participants"
            )
            if len(participants) < 2:
                raise ReviewError(
                    f"{prefix}.participants must contain at least two entries"
                )
            participant_names: set[str] = set()
            for participant_index, participant in enumerate(participants):
                value = require_string(
                    participant, f"{prefix}.participants[{participant_index}]"
                )
                if value in participant_names:
                    raise ReviewError(
                        f"{prefix}.participants contains duplicate name {value!r}"
                    )
                participant_names.add(value)
            events = require_list(diagram.get("events"), f"{prefix}.events")
            if not events:
                raise ReviewError(f"{prefix}.events must not be empty")
            for event_index, event in enumerate(events):
                event_prefix = f"{prefix}.events[{event_index}]"
                if not isinstance(event, dict):
                    raise ReviewError(f"{event_prefix} must be an object")
                source = require_string(event.get("from"), f"{event_prefix}.from")
                destination = require_string(event.get("to"), f"{event_prefix}.to")
                if (
                    source not in participant_names
                    or destination not in participant_names
                ):
                    raise ReviewError(
                        f"{event_prefix} references an unknown participant"
                    )
                if source == destination:
                    raise ReviewError(
                        f"{event_prefix} must connect two different participants"
                    )
                require_string(event.get("label"), f"{event_prefix}.label")
                require_string(
                    event.get("detail", ""), f"{event_prefix}.detail", allow_empty=True
                )
                tone = event.get("tone", "neutral")
                if tone not in VALID_DIAGRAM_TONES:
                    raise ReviewError(
                        f"{event_prefix}.tone must be one of {sorted(VALID_DIAGRAM_TONES)}"
                    )
                target = event.get("target")
                if target is not None and target not in ids:
                    raise ReviewError(
                        f"{event_prefix}.target references unknown id {target!r}"
                    )

    manifest_coverage: set[str] = set()
    for index, manifest in enumerate(manifests):
        prefix = f"manifests[{index}]"
        require_string(manifest.get("title"), f"{prefix}.title")
        require_string(
            manifest.get("description", ""), f"{prefix}.description", allow_empty=True
        )
        require_string(manifest.get("language", "yaml"), f"{prefix}.language")
        require_string(manifest.get("code"), f"{prefix}.code")
        related = validate_finding_ids(
            manifest.get("finding_ids"), f"{prefix}.finding_ids", finding_ids
        )
        manifest_coverage.update(related)

    for index, finding in enumerate(findings):
        if (
            finding.get("involves_api_objects", False)
            and finding["id"] not in manifest_coverage
        ):
            raise ReviewError(
                f"findings[{index}] involves API objects but has no linked manifest example"
            )

    for index, mapped in enumerate(file_map):
        prefix = f"file_map[{index}]"
        if not isinstance(mapped, dict):
            raise ReviewError(f"{prefix} must be an object")
        require_string(mapped.get("path"), f"{prefix}.path")
        require_positive_int(mapped.get("total_lines"), f"{prefix}.total_lines")
        for marker_index, marker in enumerate(
            require_list(mapped.get("markers", []), f"{prefix}.markers")
        ):
            marker_prefix = f"{prefix}.markers[{marker_index}]"
            if not isinstance(marker, dict):
                raise ReviewError(f"{marker_prefix} must be an object")
            line = require_positive_int(marker.get("line"), f"{marker_prefix}.line")
            kind = marker.get("kind", "relevant")
            if kind not in VALID_MARKER_KINDS:
                raise ReviewError(
                    f"{marker_prefix}.kind must be one of {sorted(VALID_MARKER_KINDS)}"
                )
            target = require_string(marker.get("target"), f"{marker_prefix}.target")
            if target not in ids:
                raise ReviewError(
                    f"{marker_prefix}.target references unknown id {target!r}"
                )

    for index, row in enumerate(matrix):
        prefix = f"test_matrix[{index}]"
        if not isinstance(row, dict):
            raise ReviewError(f"{prefix} must be an object")
        validate_finding_ids(
            row.get("finding_ids"), f"{prefix}.finding_ids", finding_ids
        )
        require_string(row.get("case"), f"{prefix}.case")
        require_string(row.get("expected"), f"{prefix}.expected")
        require_string(row.get("status"), f"{prefix}.status")
        tone = row.get("tone", "neutral")
        if tone not in VALID_TEST_TONES:
            raise ReviewError(
                f"{prefix}.tone must be one of {sorted(VALID_TEST_TONES)}"
            )
        file_path = require_string(row.get("file"), f"{prefix}.file")
        side = require_string(row.get("side"), f"{prefix}.side")
        if side not in VALID_SIDES:
            raise ReviewError(f"{prefix}.side must be old or new")
        line = require_positive_int(row.get("line"), f"{prefix}.line")
        if (file_path, side, line) not in diff["line_keys"]:
            raise ReviewError(
                f"{prefix} points outside the diff: {file_path}:{side}:{line}"
            )

    for index, reference in enumerate(references):
        prefix = f"references[{index}]"
        validate_finding_ids(
            reference.get("finding_ids"), f"{prefix}.finding_ids", finding_ids
        )
        require_string(reference.get("path"), f"{prefix}.path")
        require_string(reference.get("summary"), f"{prefix}.summary")
        code = require_string(reference.get("code"), f"{prefix}.code")
        code_lines = code.split("\n")
        highlighted = require_list(
            reference.get("highlight_lines"), f"{prefix}.highlight_lines"
        )
        if not highlighted:
            raise ReviewError(
                f"{prefix}.highlight_lines must contain at least one line"
            )
        seen_highlights: set[int] = set()
        for highlight_index, line in enumerate(highlighted):
            line = require_positive_int(
                line, f"{prefix}.highlight_lines[{highlight_index}]"
            )
            if line > len(code_lines):
                raise ReviewError(
                    f"{prefix}.highlight_lines[{highlight_index}] points past the {len(code_lines)}-line snippet"
                )
            if line in seen_highlights:
                raise ReviewError(
                    f"{prefix}.highlight_lines contains duplicate line {line}"
                )
            seen_highlights.add(line)
        if min(seen_highlights) == 1 or max(seen_highlights) == len(code_lines):
            raise ReviewError(
                f"{prefix}.code must include context before and after highlighted lines"
            )
        back_target = reference.get("back_target")
        if back_target is not None and back_target not in ids:
            raise ReviewError(
                f"{prefix}.back_target references unknown id {back_target!r}"
            )

    for index, block in enumerate(code_blocks):
        prefix = f"code_blocks[{index}]"
        validate_finding_ids(
            block.get("finding_ids"), f"{prefix}.finding_ids", finding_ids
        )
        file_path = require_string(block.get("file"), f"{prefix}.file")
        side = require_string(block.get("side"), f"{prefix}.side")
        if side not in VALID_SIDES:
            raise ReviewError(f"{prefix}.side must be old or new")
        start_line = block.get("start_line")
        end_line = block.get("end_line")
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or start_line < 1
        ):
            raise ReviewError(f"{prefix}.start_line must be a positive integer")
        if (
            not isinstance(end_line, int)
            or isinstance(end_line, bool)
            or end_line < start_line
        ):
            raise ReviewError(
                f"{prefix}.end_line must be an integer at or after start_line"
            )
        if (file_path, side, start_line) not in diff["line_keys"]:
            raise ReviewError(
                f"{prefix}.start_line points outside the diff: {file_path}:{side}:{start_line}"
            )
        if (file_path, side, end_line) not in diff["line_keys"]:
            raise ReviewError(
                f"{prefix}.end_line points outside the diff: {file_path}:{side}:{end_line}"
            )
        heat = require_string(block.get("heat"), f"{prefix}.heat")
        if heat not in VALID_HEAT_LEVELS:
            raise ReviewError(
                f"{prefix}.heat must be one of {sorted(VALID_HEAT_LEVELS)}"
            )
        smell = require_string(block.get("smell"), f"{prefix}.smell")
        if smell not in VALID_SMELL_LEVELS:
            raise ReviewError(
                f"{prefix}.smell must be one of {sorted(VALID_SMELL_LEVELS)}"
            )
        smell_reason = require_string(
            block.get("smell_reason", ""), f"{prefix}.smell_reason", allow_empty=True
        )
        if smell == "none" and smell_reason:
            raise ReviewError(f"{prefix}.smell_reason must be empty when smell is none")
        if smell != "none" and not smell_reason:
            raise ReviewError(
                f"{prefix}.smell_reason is required when smell is {smell}"
            )
        require_string(block.get("summary"), f"{prefix}.summary")

    normalized = dict(spec)
    normalized.setdefault("version", 1)
    normalized.setdefault("subtitle", "")
    normalized.setdefault("source_url", "")
    normalized.setdefault("revision", "")
    normalized.setdefault("status", f"{len(findings)} findings")
    normalized.setdefault("summary", "")
    normalized.setdefault("scope", {"in": [], "out": []})
    if normalized.get("agent_prompt"):
        raise ReviewError(
            "top-level agent_prompt is not supported; put agent_prompt on each finding"
        )
    normalized.setdefault("validation_note", "")
    normalized.setdefault("code_blocks", [])
    normalized.setdefault("diagrams", [])
    normalized.setdefault("manifests", [])
    normalized.setdefault("github_collapsed_files", [])
    github_files_url = github_pr_files_url(normalized["source_url"])
    gitlab_files_url = gitlab_mr_diffs_url(normalized["source_url"])
    github_anchors = github_diff_anchors(normalized["source_url"], diff["files"])
    gitlab_anchors = gitlab_diff_anchors(normalized["source_url"], diff["files"])
    source_provider = ""
    source_diff_anchors: dict[str, str] = {}
    if github_files_url:
        source_provider = "github"
        source_diff_anchors = github_anchors
    elif gitlab_files_url:
        source_provider = "gitlab"
        source_diff_anchors = gitlab_anchors
    normalized["derived"] = {
        "changed_files": diff["changed_files"],
        "additions": diff["additions"],
        "deletions": diff["deletions"],
        "files": diff["files"],
        "source_provider": source_provider,
        "source_files_url": github_files_url or gitlab_files_url,
        "source_diff_anchors": source_diff_anchors,
        "source_line_anchors": gitlab_line_anchors(normalized["source_url"], diff),
        "github_files_url": github_files_url,
        "github_diff_anchors": github_anchors,
    }
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, required=True, help="review JSON specification"
    )
    parser.add_argument("--diff", type=Path, required=True, help="unified diff")
    parser.add_argument("--output", type=Path, help="output HTML path")
    parser.add_argument("--template", type=Path, help="override HTML template")
    parser.add_argument(
        "--validate-only", action="store_true", help="validate without writing HTML"
    )
    args = parser.parse_args()

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        raw_diff = args.diff.read_text(encoding="utf-8")
        parsed = parse_diff(raw_diff)
        normalized = validate_spec(spec, parsed)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "changed_files": parsed["changed_files"],
                        "additions": parsed["additions"],
                        "deletions": parsed["deletions"],
                        "findings": len(normalized["findings"]),
                    }
                )
            )
            return 0
        if args.output is None:
            raise ReviewError("--output is required unless --validate-only is used")
        asset_root = Path(__file__).resolve().parent.parent / "assets"
        template = args.template or asset_root / "review-template.html"
        html = template.read_text(encoding="utf-8")
        payload = dict(normalized)
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).replace("</", "<\\/")
        replacements = {
            "__CYTOSCAPE_JS__": (
                "/*\n"
                + (asset_root / "vendor" / "cytoscape.LICENSE").read_text(
                    encoding="utf-8"
                )
                + "\n*/\n"
                + (asset_root / "vendor" / "cytoscape-3.33.4.min.js").read_text(
                    encoding="utf-8"
                )
            ).replace("</", "<\\/"),
            "__DAGRE_JS__": (
                "/*\n"
                + (asset_root / "vendor" / "dagre.LICENSE").read_text(encoding="utf-8")
                + "\n*/\n"
                + (asset_root / "vendor" / "dagre-0.8.5.min.js").read_text(
                    encoding="utf-8"
                )
            ).replace("</", "<\\/"),
            "__CYTOSCAPE_DAGRE_JS__": (
                "/*\n"
                + (asset_root / "vendor" / "cytoscape-dagre.LICENSE").read_text(
                    encoding="utf-8"
                )
                + "\n*/\n"
                + (asset_root / "vendor" / "cytoscape-dagre-2.5.0.js").read_text(
                    encoding="utf-8"
                )
            ).replace("</", "<\\/"),
            "__REVIEW_DATA_JSON__": encoded,
        }
        rendered = html
        for placeholder, value in replacements.items():
            if rendered.count(placeholder) != 1:
                raise ReviewError(
                    f"template must contain exactly one {placeholder} placeholder"
                )
            rendered = rendered.replace(placeholder, value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(
            f"wrote {args.output} ({parsed['changed_files']} files, {len(normalized['findings'])} findings)"
        )
        return 0
    except (OSError, json.JSONDecodeError, ReviewError) as exc:
        print(f"visual-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

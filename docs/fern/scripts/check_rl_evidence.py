#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit the source and external evidence behind the RL documentation.

The default audit is deterministic and network-free, so it can run in docs CI.
It verifies that every evidence-ledger record has a machine-readable rule, the
reviewed source contracts still contain their load-bearing symbols, package
pins and maturity labels agree with the docs, and the framework matrix has one
authoritative location.

Release maintainers can add ``--online`` to compare recorded GitHub branch and
pull-request states with the live API and inspect load-bearing files at pinned
framework commits. ``--release`` detects changes in watched Dynamo subsystems
since the recorded baseline commit.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

FERN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FERN_ROOT.parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("rl_evidence.json")
LEDGER_ID = re.compile(r"^\|\s*(RL-[A-Z0-9-]+)\s*\|", re.MULTILINE)
FENCED_BLOCK = re.compile(
    r"^(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n(?P<body>.*?)^(?P=fence)\s*$",
    re.MULTILINE | re.DOTALL,
)
SHA256 = re.compile(r"[0-9a-f]{64}")
CONTRACT_TOKEN_PATTERNS = {
    "environment": re.compile(r"\b(?:DYN|VERL|CUDA|NCCL|NIXL|RAY|GITHUB)_[A-Z0-9_]+\b"),
    "option": re.compile(r"(?<![\w])--[a-z0-9][a-z0-9-]*"),
    "route": re.compile(
        r"/(?:inference/v1|v1|engine|generate|metrics|live|health)"
        r"(?:/[A-Za-z0-9_.{}*:-]+)*"
    ),
    "header": re.compile(r"\bX-[A-Za-z0-9-]+\b", re.IGNORECASE),
    "field": re.compile(
        r"\b(?:nvext|sampling_params|agent_hints|request_end|request_payload|payload|"
        r"framework|target_version|record_state|routing|weight_paths|observability|"
        r"replay_simulation|owners|run_window|pins|gates)\.[A-Za-z0-9_.]+\b"
    ),
}
URL_PORT = re.compile(r"https?://[^\s`\"']+?:(\d{2,5})(?=[/\s`\"'])")
BACKTICK_IDENTIFIER = re.compile(r"`([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)`")
NON_CONTRACT_IDENTIFIERS = {
    "abort",
    "etcd",
    "fcfs",
    "git",
    "keep",
    "kv",
    "latest",
    "live_measurement",
    "main",
    "not_run",
    "null",
    "nvext",
    "passed",
    "planned",
    "random",
    "verl",
    "wait",
    "wspt",
}


class ManifestError(ValueError):
    """Raised when the audit manifest cannot be interpreted safely."""


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return payload


def _repo_path(repo: Path, raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(f"manifest path must be repository-relative: {raw!r}")
    root = repo.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ManifestError(f"manifest path escapes the repository: {raw!r}")
    return candidate


def _read(repo: Path, raw: str, findings: list[str]) -> str | None:
    try:
        path = _repo_path(repo, raw)
    except ManifestError as exc:
        findings.append(str(exc))
        return None
    if not path.is_file():
        findings.append(f"{raw}: expected file does not exist")
        return None
    return path.read_text(encoding="utf-8")


def _strings(value: Any, field: str, record_id: str, findings: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        findings.append(f"{record_id}: {field} must be a list of strings")
        return []
    return value


def _check_file_assertion(
    repo: Path,
    assertion: dict[str, Any],
    record_id: str,
    findings: list[str],
) -> None:
    raw_path = assertion.get("path")
    if not isinstance(raw_path, str):
        findings.append(f"{record_id}: assertion path must be a string")
        return
    text = _read(repo, raw_path, findings)
    if text is None:
        return
    _check_text_assertion(text, raw_path, assertion, record_id, findings)


def _check_text_assertion(
    text: str,
    label: str,
    assertion: dict[str, Any],
    record_id: str,
    findings: list[str],
) -> None:
    required = _strings(assertion.get("contains"), "contains", record_id, findings)
    forbidden = _strings(
        assertion.get("not_contains"), "not_contains", record_id, findings
    )
    if not required and not forbidden:
        findings.append(f"{record_id}: {label} assertion has no checks")
    for needle in required:
        if needle not in text:
            findings.append(f"{record_id}: {label} no longer contains {needle!r}")
    for needle in forbidden:
        if needle in text:
            findings.append(
                f"{record_id}: {label} now contains forbidden expiration trigger {needle!r}"
            )


def _record_ids(
    records: Iterable[dict[str, Any]], section: str, findings: list[str]
) -> list[str]:
    result: list[str] = []
    for index, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.startswith("RL-"):
            findings.append(f"{section}[{index}]: id must be an RL-* string")
            continue
        result.append(record_id)
    return result


def _snippet_blocks(text: str, excluded_languages: set[str]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in FENCED_BLOCK.finditer(text):
        language = match.group("info").strip()
        if language in excluded_languages:
            continue
        content = match.group("body").rstrip("\n")
        digest = hashlib.sha256(f"{language}\n{content}".encode()).hexdigest()
        blocks.append((language, digest))
    return blocks


def _contract_tokens(text: str) -> dict[str, set[str]]:
    tokens = {
        category: set(pattern.findall(text))
        for category, pattern in CONTRACT_TOKEN_PATTERNS.items()
    }
    tokens["header"] = {token.lower() for token in tokens["header"]}
    tokens["field"] = {
        token
        for token in tokens["field"]
        if not token.endswith((".md", ".jsonl", ".v1"))
    }
    tokens["field"].update(
        token
        for token in BACKTICK_IDENTIFIER.findall(text)
        if token not in NON_CONTRACT_IDENTIFIERS
        and not token.endswith((".md", ".toml", ".sh"))
        and not re.fullmatch(r"[0-9a-f]{7,40}", token)
    )
    tokens["port"] = set(URL_PORT.findall(text))
    return tokens


def _check_contract_surface(
    manifest: dict[str, Any],
    repo: Path,
    valid_evidence_ids: set[str],
    findings: list[str],
) -> None:
    surface = manifest.get("contract_surface")
    if not isinstance(surface, dict):
        findings.append("contract_surface must be an object")
        return
    pattern = surface.get("glob")
    if not isinstance(pattern, str) or not pattern:
        findings.append("contract_surface.glob must be a non-empty string")
        return
    relative_pattern = Path(pattern)
    if relative_pattern.is_absolute() or ".." in relative_pattern.parts:
        findings.append("contract_surface.glob must be repository-relative")
        return
    actual = {category: set() for category in (*CONTRACT_TOKEN_PATTERNS, "port")}
    for path in sorted(repo.glob(pattern)):
        if not path.is_file():
            continue
        for category, tokens in _contract_tokens(
            path.read_text(encoding="utf-8")
        ).items():
            actual[category].update(tokens)

    records = surface.get("records")
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        findings.append("contract_surface.records must be a list of objects")
        return
    declared = {category: set() for category in actual}
    for index, record in enumerate(records):
        label = f"contract_surface.records[{index}]"
        category = record.get("category")
        if category not in actual:
            findings.append(
                f"{label}.category must be one of {', '.join(sorted(actual))}"
            )
            continue
        record_tokens = _strings(record.get("tokens"), "tokens", label, findings)
        if not record_tokens:
            findings.append(f"{label}.tokens must not be empty")
        if record_tokens != sorted(set(record_tokens)):
            findings.append(f"{label}.tokens must be sorted and unique")
        overlap = declared[category].intersection(record_tokens)
        if overlap:
            findings.append(
                f"{label} redeclares {category} tokens: {', '.join(sorted(overlap))}"
            )
        declared[category].update(record_tokens)
        evidence_ids = _strings(
            record.get("evidence_ids"), "evidence_ids", label, findings
        )
        if not evidence_ids:
            findings.append(f"{label}.evidence_ids must not be empty")
        for evidence_id in evidence_ids:
            if evidence_id not in valid_evidence_ids:
                findings.append(f"{label} references unknown evidence ID {evidence_id}")
        for field in ("owner", "expiration_trigger"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                findings.append(f"{label}.{field} must be a non-empty string")

    for category in actual:
        missing = sorted(actual[category] - declared[category])
        stale = sorted(declared[category] - actual[category])
        if missing:
            findings.append(
                f"unrecorded {category} contract tokens: {', '.join(missing)}"
            )
        if stale:
            findings.append(f"stale {category} contract tokens: {', '.join(stale)}")


def _check_snippet_coverage(
    manifest: dict[str, Any],
    repo: Path,
    valid_evidence_ids: set[str],
    findings: list[str],
) -> None:
    coverage = manifest.get("snippet_coverage")
    if not isinstance(coverage, dict):
        findings.append("snippet_coverage must be an object")
        return
    pattern = coverage.get("glob")
    if not isinstance(pattern, str) or not pattern:
        findings.append("snippet_coverage.glob must be a non-empty string")
        return
    relative_pattern = Path(pattern)
    if relative_pattern.is_absolute() or ".." in relative_pattern.parts:
        findings.append("snippet_coverage.glob must be repository-relative")
        return
    excluded = _strings(
        coverage.get("exclude_languages"),
        "snippet_coverage.exclude_languages",
        "snippet coverage",
        findings,
    )
    excluded_languages = set(excluded)
    actual: dict[tuple[str, int], tuple[str, str]] = {}
    actual_pages: set[str] = set()
    for path in sorted(repo.glob(pattern)):
        if not path.is_file():
            continue
        blocks = _snippet_blocks(path.read_text(encoding="utf-8"), excluded_languages)
        if not blocks:
            continue
        raw_path = str(path.relative_to(repo))
        actual_pages.add(raw_path)
        for block, details in enumerate(blocks, start=1):
            actual[(raw_path, block)] = details

    pages = coverage.get("pages")
    if not isinstance(pages, list) or any(not isinstance(item, dict) for item in pages):
        findings.append("snippet_coverage.pages must be a list of objects")
        return
    declared: dict[tuple[str, int], dict[str, Any]] = {}
    declared_pages: set[str] = set()
    for index, page in enumerate(pages):
        label = f"snippet_coverage.pages[{index}]"
        raw_path = page.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            findings.append(f"{label}.path must be a non-empty string")
            continue
        try:
            _repo_path(repo, raw_path)
        except ManifestError as exc:
            findings.append(str(exc))
            continue
        if raw_path in declared_pages:
            findings.append(f"snippet coverage declares {raw_path} more than once")
        declared_pages.add(raw_path)
        for field in ("owner", "expiration_trigger"):
            if not isinstance(page.get(field), str) or not page[field].strip():
                findings.append(f"{label}.{field} must be a non-empty string")
        blocks = page.get("blocks")
        if not isinstance(blocks, list) or any(
            not isinstance(item, dict) for item in blocks
        ):
            findings.append(f"{label}.blocks must be a list of objects")
            continue
        for block_index, record in enumerate(blocks):
            record_label = f"{label}.blocks[{block_index}]"
            block = record.get("block")
            language = record.get("language")
            digest = record.get("sha256")
            evidence_ids = _strings(
                record.get("evidence_ids"),
                "evidence_ids",
                record_label,
                findings,
            )
            if not isinstance(block, int) or isinstance(block, bool) or block < 1:
                findings.append(f"{record_label}.block must be a positive integer")
                continue
            key = (raw_path, block)
            if key in declared:
                findings.append(
                    f"snippet coverage declares {raw_path} block {block} more than once"
                )
            declared[key] = record
            if not isinstance(language, str) or not language:
                findings.append(f"{record_label}.language must be a non-empty string")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                findings.append(
                    f"{record_label}.sha256 must be a full lowercase digest"
                )
            if not evidence_ids:
                findings.append(f"{record_label}.evidence_ids must not be empty")
            for evidence_id in evidence_ids:
                if evidence_id not in valid_evidence_ids:
                    findings.append(
                        f"{record_label} references unknown evidence ID {evidence_id}"
                    )

    missing_pages = sorted(actual_pages - declared_pages)
    extra_pages = sorted(declared_pages - actual_pages)
    if missing_pages:
        findings.append(
            "snippet pages without coverage records: " + ", ".join(missing_pages)
        )
    if extra_pages:
        findings.append(
            "snippet coverage pages without eligible blocks: " + ", ".join(extra_pages)
        )
    for key, (actual_language, actual_digest) in actual.items():
        raw_path, block = key
        record = declared.get(key)
        if record is None:
            findings.append(f"{raw_path} block {block} has no snippet evidence record")
            continue
        if record.get("language") != actual_language:
            findings.append(
                f"{raw_path} block {block} language changed from {record.get('language')!r} to {actual_language!r}"
            )
        if record.get("sha256") != actual_digest:
            findings.append(
                f"{raw_path} block {block} content changed; review its command/config evidence and record sha256 {actual_digest}"
            )
    for raw_path, block in sorted(set(declared) - set(actual)):
        findings.append(
            f"snippet evidence record has no matching {raw_path} block {block}"
        )


def _git_changed_paths(
    repo: Path, baseline: str, watched: list[str]
) -> tuple[list[str], str | None]:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{baseline}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        return (
            [],
            f"baseline commit {baseline} is unavailable; fetch history before the release audit",
        )
    command = ["git", "diff", "--name-only", baseline, "--", *watched]
    diff = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode:
        return (
            [],
            f"cannot compare watched paths with {baseline}: {diff.stderr.strip()}",
        )
    return sorted(path for path in diff.stdout.splitlines() if path), None


def check_offline(
    manifest: dict[str, Any],
    repo: Path,
    *,
    release: bool = False,
    max_age_days: int | None = None,
    today: date | None = None,
) -> list[str]:
    findings: list[str] = []
    if manifest.get("schema_version") != 2:
        findings.append("schema_version must be 2")

    ledger_path = manifest.get("ledger")
    if not isinstance(ledger_path, str):
        findings.append("ledger must be a repository-relative path")
        ledger_text = None
    else:
        ledger_text = _read(repo, ledger_path, findings)

    claims = manifest.get("claims", [])
    github = manifest.get("github", {})
    branches = github.get("branches", []) if isinstance(github, dict) else []
    pulls = github.get("pulls", []) if isinstance(github, dict) else []
    sections = (
        ("claims", claims),
        ("github.branches", branches),
        ("github.pulls", pulls),
    )
    records: list[dict[str, Any]] = []
    ids: list[str] = []
    for section, value in sections:
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            findings.append(f"{section} must be a list of objects")
            continue
        records.extend(value)
        ids.extend(_record_ids(value, section, findings))

    duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    if duplicates:
        findings.append(
            f"duplicate machine-readable record IDs: {', '.join(duplicates)}"
        )
    if ledger_text is not None:
        ledger_ids = set(LEDGER_ID.findall(ledger_text))
        manifest_ids = set(ids)
        missing_rules = sorted(ledger_ids - manifest_ids)
        missing_ledger = sorted(manifest_ids - ledger_ids)
        if missing_rules:
            findings.append(
                f"ledger records without audit rules: {', '.join(missing_rules)}"
            )
        if missing_ledger:
            findings.append(
                f"audit records absent from ledger: {', '.join(missing_ledger)}"
            )

    _check_snippet_coverage(manifest, repo, set(ids), findings)
    _check_contract_surface(manifest, repo, set(ids), findings)

    for record in records:
        record_id = record.get("id", "unknown record")
        for field in ("sources", "documents"):
            assertions = record.get(field, [])
            if not isinstance(assertions, list) or any(
                not isinstance(item, dict) for item in assertions
            ):
                findings.append(f"{record_id}: {field} must be a list of objects")
                continue
            for assertion in assertions:
                _check_file_assertion(repo, assertion, record_id, findings)
        external_files = record.get("files", [])
        if not isinstance(external_files, list) or any(
            not isinstance(item, dict) for item in external_files
        ):
            findings.append(f"{record_id}: files must be a list of objects")
        else:
            for assertion in external_files:
                external_path = assertion.get("path")
                if not isinstance(external_path, str) or not external_path:
                    findings.append(
                        f"{record_id}: external file path must be a non-empty string"
                    )
                required = _strings(
                    assertion.get("contains"), "contains", record_id, findings
                )
                forbidden = _strings(
                    assertion.get("not_contains"),
                    "not_contains",
                    record_id,
                    findings,
                )
                if not required and not forbidden:
                    findings.append(
                        f"{record_id}: external file {external_path!r} has no checks"
                    )
        if not record.get("sources") and not record.get("documents"):
            findings.append(
                f"{record_id}: record has no source or documentation assertions"
            )

    pins = manifest.get("package_pins", [])
    if not isinstance(pins, list) or any(not isinstance(item, dict) for item in pins):
        findings.append("package_pins must be a list of objects")
    else:
        for pin in pins:
            name = pin.get("name", "unnamed package pin")
            source = {
                "path": pin.get("source_path"),
                "contains": [pin.get("source_value")],
            }
            if not isinstance(pin.get("source_value"), str):
                findings.append(f"{name}: source_value must be a string")
            else:
                _check_file_assertion(repo, source, str(name), findings)
            document_value = pin.get("document_value")
            documents = pin.get("documents", [])
            if not isinstance(document_value, str):
                findings.append(f"{name}: document_value must be a string")
            elif not isinstance(documents, list) or any(
                not isinstance(path, str) for path in documents
            ):
                findings.append(f"{name}: documents must be a list of paths")
            else:
                for path in documents:
                    _check_file_assertion(
                        repo,
                        {"path": path, "contains": [document_value]},
                        str(name),
                        findings,
                    )

    singleton_rules = manifest.get("singleton_markers", [])
    if not isinstance(singleton_rules, list) or any(
        not isinstance(item, dict) for item in singleton_rules
    ):
        findings.append("singleton_markers must be a list of objects")
    else:
        for rule in singleton_rules:
            pattern = rule.get("glob")
            marker = rule.get("marker")
            allowed = rule.get("allowed_paths")
            if not isinstance(pattern, str) or not isinstance(marker, str):
                findings.append("singleton marker glob and marker must be strings")
                continue
            if not isinstance(allowed, list) or any(
                not isinstance(path, str) for path in allowed
            ):
                findings.append(
                    f"singleton marker {marker!r}: allowed_paths must be a list"
                )
                continue
            actual = {
                str(path.relative_to(repo))
                for path in repo.glob(pattern)
                if path.is_file() and marker in path.read_text(encoding="utf-8")
            }
            if actual != set(allowed):
                findings.append(
                    f"singleton marker {marker!r} appears in {sorted(actual)}, expected {sorted(allowed)}"
                )

    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        findings.append("baseline must be an object")
    else:
        commit = baseline.get("dynamo_commit")
        documents = baseline.get("documents", [])
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            findings.append("baseline.dynamo_commit must be a full lowercase SHA")
        elif not isinstance(documents, list) or any(
            not isinstance(path, str) for path in documents
        ):
            findings.append("baseline.documents must be a list of paths")
        else:
            for path in documents:
                _check_file_assertion(
                    repo,
                    {"path": path, "contains": [commit]},
                    "Dynamo baseline",
                    findings,
                )
            if release:
                watched = baseline.get("watch_paths", [])
                if not isinstance(watched, list) or any(
                    not isinstance(path, str) for path in watched
                ):
                    findings.append("baseline.watch_paths must be a list of paths")
                elif not watched:
                    findings.append(
                        "baseline.watch_paths cannot be empty in release mode"
                    )
                else:
                    changed, error = _git_changed_paths(repo, commit, watched)
                    if error:
                        findings.append(error)
                    if changed:
                        findings.append(
                            "reviewed Dynamo baseline expired; re-audit these watched paths and update the pin: "
                            + ", ".join(changed)
                        )

    if max_age_days is not None:
        reviewed = manifest.get("reviewed_on")
        try:
            reviewed_date = date.fromisoformat(reviewed)
        except (TypeError, ValueError):
            findings.append(
                "reviewed_on must be an ISO date when --max-age-days is used"
            )
        else:
            age = ((today or date.today()) - reviewed_date).days
            if age < 0:
                findings.append(f"reviewed_on {reviewed} is in the future")
            elif age > max_age_days:
                findings.append(
                    f"evidence review is {age} days old, exceeding --max-age-days {max_age_days}"
                )
    return findings


def github_fetcher(
    *,
    api_base: str = "https://api.github.com",
    token: str | None = None,
    timeout: float = 15.0,
) -> Callable[[str], dict[str, Any]]:
    base = api_base.rstrip("/")

    def fetch(path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            base + path,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "dynamo-rl-evidence-audit",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"GitHub request failed for {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub returned a non-object for {path}")
        return payload

    return fetch


def _github_file_text(
    payload: dict[str, Any], label: str
) -> tuple[str | None, str | None]:
    if payload.get("type") != "file":
        return None, f"{label}: GitHub contents response is not a file"
    if payload.get("encoding") != "base64" or not isinstance(
        payload.get("content"), str
    ):
        return None, f"{label}: GitHub contents response has no base64 content"
    encoded = "".join(payload["content"].split())
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        return None, f"{label}: cannot decode GitHub file content: {exc}"
    return decoded, None


def check_online(
    manifest: dict[str, Any], fetch: Callable[[str], dict[str, Any]]
) -> list[str]:
    findings: list[str] = []
    github = manifest.get("github", {})
    if not isinstance(github, dict):
        return ["github must be an object"]
    for branch in github.get("branches", []):
        record_id = branch.get("id", "unknown branch")
        repo = branch.get("repo")
        ref = branch.get("ref")
        expected = branch.get("expected_sha")
        if not all(isinstance(value, str) for value in (repo, ref, expected)):
            findings.append(f"{record_id}: repo, ref, and expected_sha must be strings")
            continue
        path = f"/repos/{repo}/commits/{urllib.parse.quote(ref, safe='')}"
        try:
            payload = fetch(path)
        except RuntimeError as exc:
            findings.append(f"{record_id}: {exc}")
        else:
            actual = payload.get("sha")
            if actual != expected:
                findings.append(
                    f"{record_id}: {repo}@{ref} moved from {expected} to {actual}; recheck the integration evidence"
                )
        for assertion in branch.get("files", []):
            external_path = assertion.get("path")
            if not isinstance(external_path, str):
                findings.append(f"{record_id}: external file path must be a string")
                continue
            quoted_path = urllib.parse.quote(external_path, safe="/")
            query = urllib.parse.urlencode({"ref": expected})
            label = f"{repo}@{expected}:{external_path}"
            try:
                file_payload = fetch(f"/repos/{repo}/contents/{quoted_path}?{query}")
            except RuntimeError as exc:
                findings.append(f"{record_id}: {exc}")
                continue
            file_text, error = _github_file_text(file_payload, label)
            if error:
                findings.append(f"{record_id}: {error}")
                continue
            assert file_text is not None
            _check_text_assertion(
                file_text,
                label,
                assertion,
                record_id,
                findings,
            )

    for pull in github.get("pulls", []):
        record_id = pull.get("id", "unknown pull request")
        repo = pull.get("repo")
        number = pull.get("number")
        if not isinstance(repo, str) or not isinstance(number, int):
            findings.append(f"{record_id}: repo must be a string and number an integer")
            continue
        try:
            payload = fetch(f"/repos/{repo}/pulls/{number}")
        except RuntimeError as exc:
            findings.append(f"{record_id}: {exc}")
            continue
        expected_state = pull.get("expected_state")
        expected_draft = pull.get("expected_draft")
        expected_merged = pull.get("expected_merged")
        expected_head = pull.get("expected_head_prefix")
        actual_head = (
            payload.get("head", {}).get("sha")
            if isinstance(payload.get("head"), dict)
            else None
        )
        checks = (
            ("state", expected_state, payload.get("state")),
            ("draft", expected_draft, payload.get("draft")),
            ("merged", expected_merged, payload.get("merged_at") is not None),
        )
        for label, expected, actual in checks:
            if actual != expected:
                findings.append(
                    f"{record_id}: {repo}#{number} {label} is {actual!r}, recorded as {expected!r}"
                )
        if not isinstance(expected_head, str) or not isinstance(actual_head, str):
            findings.append(f"{record_id}: cannot compare the recorded PR head")
        elif not actual_head.startswith(expected_head):
            findings.append(
                f"{record_id}: {repo}#{number} head moved from {expected_head} to {actual_head}"
            )
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--online",
        action="store_true",
        help="check recorded GitHub refs, PR states, and pinned framework files",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="check watched source drift since the baseline",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        help="fail when reviewed_on is older than this many days",
    )
    parser.add_argument("--github-api", default="https://api.github.com")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.max_age_days is not None and args.max_age_days < 0:
        parser.error("--max-age-days must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    findings = check_offline(
        manifest,
        args.repo,
        release=args.release,
        max_age_days=args.max_age_days,
    )
    if args.online:
        fetch = github_fetcher(
            api_base=args.github_api,
            token=os.environ.get("GITHUB_TOKEN"),
            timeout=args.timeout,
        )
        findings.extend(check_online(manifest, fetch))
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    modes = ["offline"]
    if args.online:
        modes.append("GitHub")
    if args.release:
        modes.append("release-drift")
    snippet_pages = manifest["snippet_coverage"]["pages"]
    snippet_count = sum(len(page["blocks"]) for page in snippet_pages)
    contract_token_count = sum(
        len(record["tokens"]) for record in manifest["contract_surface"]["records"]
    )
    print(
        f"RL evidence audit passed ({', '.join(modes)}; "
        f"{len(manifest.get('claims', []))} claims; {snippet_count} snippets; "
        f"{contract_token_count} contract tokens)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

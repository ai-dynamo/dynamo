#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and auto-upgrade release dependency pins.

.github/release-manifest.yaml declares the exact pin values the next
release must ship with (framework runtime image tags, NIXL refs, infra
versions in container/context.yaml). The release manager reviews and
updates the manifest during release planning; this script enforces it
at cut time (release-branch-guard.yml) and publish time (release.yml).

Modes:
  --check   Gate a release-branch cut. A pin whose current value differs
            from `required` is a violation. Auto-upgradable violations
            (auto_upgrade: true) still exit 0 so the caller can run
            --apply; anything else blocks. With --strict, ANY mismatch
            blocks (publish gate: nothing is going to apply an upgrade
            there, so drift must not ship). With --version X.Y.Z, also
            require the manifest's release.version to equal X.Y.Z.
  --apply   Rewrite auto-upgradable stale pins to `required`, preserving
            comments and quoting (raw line edit + YAML round-trip
            verification). Prints the repo-relative paths of changed
            files to stdout, one per line; diagnostics go to stderr.
            Idempotent. Refuses to run if blocking violations exist.
  --lint    Structural check for PRs: the manifest parses and every pin
            path resolves in the current tree. No value comparison
            (main legitimately drifts between releases).

Values are compared as exact strings; there is deliberately no version
ordering. If a pin was intentionally upgraded past the manifest, update
the manifest (one line) rather than expecting a >= comparison.

Exit codes (container/compliance/base_sboms/check_drift.py convention):
  0  pass (for --check: the cut/publish may proceed)
  1  violations (blocked cut/publish, or lint failures)
  2  infra/self error (malformed manifest, missing file, unresolvable
     path) - callers must not treat this as a pass
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import yaml

DEFAULT_MANIFEST = Path(".github/release-manifest.yaml")
_RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
# A mapping-key line: captures indent and key. Excludes comments and
# list items; the key stops at the first ':' so URL-ish values are fine.
_KEY_LINE_RE = re.compile(r"^(\s*)([^\s#-][^:#]*):(.*)$")


class ManifestError(Exception):
    """Exit-2 class problem: the manifest or the tree is unusable."""


@dataclasses.dataclass
class Pin:
    name: str
    file: str
    path: str
    required: str
    auto_upgrade: bool


@dataclasses.dataclass
class PinStatus:
    pin: Pin
    actual: str
    keys: list[str]  # concrete key path resolved against the file

    @property
    def ok(self) -> bool:
        return self.actual == self.pin.required


def load_manifest(path: Path) -> tuple[str, list[Pin]]:
    """Parse and schema-validate the manifest. Returns (version, pins)."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"manifest {path} is not a mapping")
    if data.get("schema_version") != 1:
        raise ManifestError("unsupported schema_version (expected 1)")
    version = (data.get("release") or {}).get("version")
    if not isinstance(version, str) or not _RELEASE_VERSION_RE.match(version):
        raise ManifestError("release.version must be an X.Y.Z string")
    raw = data.get("pins")
    if not isinstance(raw, list) or not raw:
        raise ManifestError("pins must be a non-empty list")
    pins: list[Pin] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ManifestError(f"pins[{i}] is not a mapping")
        missing = [k for k in ("name", "file", "path", "required") if not entry.get(k)]
        if missing or not isinstance(entry.get("auto_upgrade"), bool):
            raise ManifestError(
                f"pins[{i}] ({entry.get('name', '?')}): needs non-empty "
                f"name/file/path/required and boolean auto_upgrade"
            )
        if entry["name"] in seen:
            raise ManifestError(f"duplicate pin name: {entry['name']}")
        seen.add(entry["name"])
        pins.append(
            Pin(
                name=str(entry["name"]),
                file=str(entry["file"]),
                path=str(entry["path"]),
                required=str(entry["required"]),
                auto_upgrade=entry["auto_upgrade"],
            )
        )
    return version, pins


def resolve_dotted(node: object, dotted: str, context: str) -> tuple[list[str], object]:
    """Resolve a dotted path whose keys may themselves contain dots.

    context.yaml has keys like 'cuda13.0', so at each mapping level the
    longest key that prefixes the remaining path wins; zero matches or a
    length tie is an error.
    """
    keys: list[str] = []
    rest = dotted
    while rest:
        if not isinstance(node, dict):
            raise ManifestError(
                f"{context}: '{dotted}' hits a non-mapping at '{'.'.join(keys)}'"
            )
        matches = [
            k
            for k in node
            if isinstance(k, str) and (rest == k or rest.startswith(k + "."))
        ]
        if not matches:
            raise ManifestError(
                f"{context}: cannot resolve '{dotted}' (stuck at '{rest}')"
            )
        if len(matches) > 1:
            raise ManifestError(
                f"{context}: ambiguous path '{dotted}' at '{rest}' "
                f"(candidates: {sorted(matches)})"
            )
        best = matches[0]
        keys.append(best)
        node = node[best]
        rest = rest[len(best) + 1 :]
    return keys, node


def set_yaml_scalar(text: str, keys: list[str], new_value: str, context: str) -> str:
    """Rewrite one scalar in raw YAML text, preserving comments/quoting.

    Walks lines with an indentation stack so identical keys at different
    depths (e.g. sglang.nixl_ref vs sglang.xpu.nixl_ref) cannot collide;
    exactly one line must match the resolved key path.
    """
    lines = text.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    hits: list[int] = []
    for idx, line in enumerate(lines):
        m = _KEY_LINE_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group(2).strip()))
        if [k for _, k in stack] == keys:
            hits.append(idx)
    if len(hits) != 1:
        raise ManifestError(
            f"{context}: found {len(hits)} lines for '{'.'.join(keys)}'; expected 1"
        )
    target = lines[hits[0]]
    m = re.match(
        r"^(?P<head>\s*[^:#]+:\s+)(?P<val>\"[^\"]*\"|'[^']*'|[^\s#]+)(?P<tail>.*\n?)$",
        target,
    )
    if not m:
        raise ManifestError(f"{context}: cannot parse scalar line: {target.rstrip()}")
    quote = m.group("val")[0] if m.group("val")[0] in "\"'" else ""
    lines[hits[0]] = f"{m.group('head')}{quote}{new_value}{quote}{m.group('tail')}"
    return "".join(lines)


def evaluate(pins: list[Pin], repo_root: Path) -> list[PinStatus]:
    """Resolve every pin against the tree. Raises ManifestError on exit-2 problems."""
    parsed: dict[str, object] = {}
    statuses: list[PinStatus] = []
    for pin in pins:
        if pin.file not in parsed:
            try:
                parsed[pin.file] = yaml.safe_load((repo_root / pin.file).read_text())
            except (OSError, yaml.YAMLError) as exc:
                raise ManifestError(
                    f"{pin.name}: cannot read {pin.file}: {exc}"
                ) from exc
        keys, value = resolve_dotted(parsed[pin.file], pin.path, pin.name)
        if isinstance(value, (dict, list)) or value is None:
            raise ManifestError(f"{pin.name}: '{pin.path}' is not a scalar")
        statuses.append(PinStatus(pin=pin, actual=str(value), keys=keys))
    return statuses


def _write_github_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        if "\n" in value:
            delim = "__RELEASE_PINS_EOF__"
            assert delim not in value
            fh.write(f"{name}<<{delim}\n{value}\n{delim}\n")
        else:
            fh.write(f"{name}={value}\n")


def _write_step_summary(markdown: str) -> None:
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")


def _report(statuses: list[PinStatus], problems: list[str], strict: bool) -> None:
    """Emit the human table (stderr), step summary, and GITHUB_OUTPUT values."""
    rows = []
    slack_lines = []
    for s in statuses:
        if s.ok:
            state = "ok"
        elif s.pin.auto_upgrade and not strict:
            state = "stale (auto-upgrade)"
        else:
            state = "BLOCKED"
        rows.append((s.pin.name, s.pin.required, s.actual, state))
        mark = ":white_check_mark:" if s.ok else ":x:"
        slack_lines.append(
            f"{mark} {s.pin.name}: required {s.pin.required}, actual {s.actual}"
        )
    width = max(len(r[0]) for r in rows)
    for r in rows:
        print(
            f"  {r[0]:<{width}}  required={r[1]}  actual={r[2]}  [{r[3]}]",
            file=sys.stderr,
        )
    for p in problems:
        print(f"  BLOCKED: {p}", file=sys.stderr)
    md = ["| pin | required | actual | state |", "|---|---|---|---|"]
    md += [f"| {a} | `{b}` | `{c}` | {d} |" for a, b, c, d in rows]
    md += [f"\n**BLOCKED**: {p}" for p in problems]
    _write_step_summary("### Release pin check\n" + "\n".join(md))
    blocked = bool(problems) or any(
        not s.ok and (strict or not s.pin.auto_upgrade) for s in statuses
    )
    _write_github_output("blocked", "true" if blocked else "false")
    _write_github_output(
        "auto_upgrades",
        json.dumps([s.pin.name for s in statuses if not s.ok and s.pin.auto_upgrade]),
    )
    _write_github_output(
        "pins",
        json.dumps(
            [
                {"name": r[0], "required": r[1], "actual": r[2], "state": r[3]}
                for r in rows
            ]
        ),
    )
    _write_github_output("pin_table", "\n".join(slack_lines))


def run_check(
    manifest: Path, repo_root: Path, cli_version: str | None, strict: bool
) -> int:
    version, pins = load_manifest(manifest)
    problems: list[str] = []
    if cli_version and cli_version != version:
        problems.append(
            f"manifest release.version is {version} but this release is "
            f"{cli_version}; review and update {manifest} during release planning"
        )
    statuses = evaluate(pins, repo_root)
    _report(statuses, problems, strict)
    blocked = [s for s in statuses if not s.ok and (strict or not s.pin.auto_upgrade)]
    return 1 if problems or blocked else 0


def run_apply(manifest: Path, repo_root: Path) -> int:
    _, pins = load_manifest(manifest)
    statuses = evaluate(pins, repo_root)
    blocked = [s for s in statuses if not s.ok and not s.pin.auto_upgrade]
    if blocked:
        for s in blocked:
            print(
                f"BLOCKED (not auto-upgradable): {s.pin.name} "
                f"required={s.pin.required} actual={s.actual}",
                file=sys.stderr,
            )
        return 1
    upgrades = [s for s in statuses if not s.ok]
    by_file: dict[str, list[PinStatus]] = {}
    for s in upgrades:
        by_file.setdefault(s.pin.file, []).append(s)
    for file, file_upgrades in sorted(by_file.items()):
        path = repo_root / file
        original = path.read_text()
        text = original
        for s in file_upgrades:
            text = set_yaml_scalar(text, s.keys, s.pin.required, s.pin.name)
        # Round-trip verification: every manifest pin in this file must now
        # read back as expected; anything off means the editor misfired.
        reparsed = yaml.safe_load(text)
        upgraded = {s.pin.name for s in file_upgrades}
        for s in statuses:
            if s.pin.file != file:
                continue
            _, value = resolve_dotted(reparsed, s.pin.path, s.pin.name)
            expected = s.pin.required if s.pin.name in upgraded else s.actual
            if str(value) != expected:
                raise ManifestError(
                    f"{s.pin.name}: post-edit verification failed in {file} "
                    f"(got {value!r}, expected {expected!r}); file left untouched"
                )
        path.write_text(text)
        print(file)
        for s in file_upgrades:
            print(f"  {s.pin.name}: {s.actual} -> {s.pin.required}", file=sys.stderr)
    return 0


def run_lint(manifest: Path, repo_root: Path) -> int:
    _, pins = load_manifest(manifest)
    failures = []
    for pin in pins:
        try:
            evaluate([pin], repo_root)
        except ManifestError as exc:
            failures.append(str(exc))
    for f in failures:
        print(f"LINT: {f}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--lint", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--version", help="release version X.Y.Z (check mode)")
    args = parser.parse_args(argv)
    try:
        if args.check:
            return run_check(args.manifest, args.repo_root, args.version, args.strict)
        if args.apply:
            return run_apply(args.manifest, args.repo_root)
        return run_lint(args.manifest, args.repo_root)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture parser outputs for every case across the parity fixtures.

Variant A schema requires `expected.{dynamo,vllm,sglang}` on every case.
This script runs the chosen impl against every fixture and either drift-
checks (default) or merges results into the YAML.

Default — DRIFT CHECK (read-only):
    python3 -m tests.parity.parser.capture_parser_outputs --impl dynamo
    python3 -m tests.parity.parser.capture_parser_outputs --impl vllm
    python3 -m tests.parity.parser.capture_parser_outputs --impl sglang

  Compares the parser's live output against the recorded `expected.<impl>`
  block. Exits non-zero on any drift. Use this as a CI gate / pre-flight
  after a `model_text` edit or an upstream parser bump.

`--merge` — POPULATE FIXTURES (destructive):
    python3 -m tests.parity.parser.capture_parser_outputs --impl dynamo --merge
    python3 -m tests.parity.parser.capture_parser_outputs --impl vllm --merge
    python3 -m tests.parity.parser.capture_parser_outputs --impl sglang --merge

  Writes the parser's output into `expected.<impl>` for cases that lack a
  block. With `--overwrite-if-exists`, also replaces existing blocks. Use
  this end-to-end when wiring up a new family (no hand-merge required).
  Parser outputs are merged like so:

    parser produces output equal to dynamo's → anchor-ref to dynamo
    parser produces output differing         → concrete `{calls, normal_text}`
    wrapper returns `UNAVAILABLE: <reason>`  → `{unavailable: <reason>}`
    parser raises (other)                    → SKIP with warning; record
                                               `{error: <substring>}` by hand

`--dump PATH` (optional, rarely needed):
    python3 -m tests.parity.parser.capture_parser_outputs --impl vllm --dump /tmp/v.json

  Also writes the raw per-case parser outputs to PATH as JSON. Useful for
  ad-hoc analysis; not part of the populate-or-check workflows.

Run inside a container where the target impl is installed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
FIXTURES = REPO_ROOT / "tests/parity/parser/fixtures"

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument(
    "--impl",
    required=True,
    choices=("dynamo", "vllm", "sglang"),
    help=(
        "Which parser implementation to run. `dynamo` uses the PyO3 binding; "
        "`vllm`/`sglang` use their in-process wrappers. Run inside a container "
        "with that impl installed."
    ),
)
_ap.add_argument(
    "--family",
    action="append",
    help=(
        "Limit capture to one parser family. May be repeated. Useful when "
        "refreshing a newly wired peer without rewriting the full fixture corpus."
    ),
)
_ap.add_argument(
    "--merge",
    action="store_true",
    help=(
        "Write parser outputs into `expected.<impl>` for cases lacking a "
        "block. Default behavior (no flag) is read-only drift check."
    ),
)
_ap.add_argument(
    "--overwrite-if-exists",
    action="store_true",
    help=(
        "With `--merge`, also overwrite cases that already have an "
        "`expected.<impl>` block. Required after an intentional parser "
        "behavior change."
    ),
)
_ap.add_argument(
    "--dump",
    metavar="PATH",
    help=(
        "Optional: also write raw per-case parser outputs to PATH as JSON. "
        "Independent of --merge."
    ),
)
_args = _ap.parse_args()
IMPL = _args.impl
MERGE_MODE = _args.merge
OVERWRITE = _args.overwrite_if_exists
FAMILY_FILTER = set(_args.family or [])

if OVERWRITE and not MERGE_MODE:
    _ap.error("--overwrite-if-exists requires --merge")


def _known_fixture_families() -> set[str]:
    families = set()
    for fp in sorted(FIXTURES.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        families.add(doc["family"])
    return families


if FAMILY_FILTER:
    known_families = _known_fixture_families()
    unknown_families = FAMILY_FILTER - known_families
    if unknown_families:
        _ap.error(
            "unknown --family value(s): "
            f"{', '.join(sorted(unknown_families))}. "
            f"Known families: {', '.join(sorted(known_families))}"
        )

wrapper = importlib.import_module(
    f"tests.parity.parser.{IMPL}"
)  # noqa: E402 — needs sys.path tweak above


# ----------------------------------------------------------------------
# Parser invocation


def serialize(call) -> dict:
    if hasattr(call, "to_dict"):
        return call.to_dict()
    return {"name": call.get("name"), "arguments": call.get("arguments")}


def normalize_output(d: dict) -> dict:
    """Canonical form for comparing parser output to a recorded YAML block."""
    return {
        "calls": d.get("calls") or [],
        "normal_text": d.get("normal_text") or "",
    }


def is_na_stub(case: dict[str, Any]) -> bool:
    """Explicit chart-only n/a stub: no parser input, no expected block."""
    return set(case) == {"description", "reason"} and all(
        isinstance(case[key], str) and case[key].strip()
        for key in ("description", "reason")
    )


def run_parser(family: str, case: dict) -> dict:
    """Run the parser wrapper. Returns {calls, normal_text, error}."""
    try:
        got = wrapper.parse(family, case["model_text"], case.get("tools"))
        return {
            "calls": [serialize(c) for c in (got.calls or [])],
            "normal_text": got.normal_text,
            "error": got.error,
        }
    except Exception as e:
        return {
            "calls": None,
            "normal_text": None,
            "error": f"PYTHON_EXC: {type(e).__name__}: {e}",
        }


# ----------------------------------------------------------------------
# Drift-check mode


def classify_recorded(case: dict, dyn_expected: dict) -> tuple[str, dict | None]:
    """Inspect the case's recorded expected.<IMPL> block.

    Returns (kind, payload):
      ("match",   <dynamo expected>)  — peer block is `*d_<case>` anchor ref
                                        OR a concrete block value-equal to
                                        the dynamo block (defensive: don't
                                        rely on PyYAML preserving anchor
                                        identity across reloads).
      ("concrete", {calls, normal_text})
      ("error",    {"error_substring": <s>})
      ("unavailable", {"reason": <s>})
      ("missing",  None)
    """
    block = case.get("expected", {}).get(IMPL)
    if block is None:
        return ("missing", None)
    if block is dyn_expected:
        return ("match", dyn_expected)
    if not isinstance(block, dict):
        return ("missing", None)
    if "unavailable" in block:
        return ("unavailable", {"reason": block["unavailable"]})
    if "error" in block:
        return ("error", {"error_substring": block["error"]})
    if "calls" in block or "normal_text" in block:
        concrete = normalize_output(block)
        if concrete == normalize_output(dyn_expected):
            return ("match", dyn_expected)
        return ("concrete", concrete)
    return ("missing", None)


def check_drift(case: dict, family: str, case_id: str, got: dict) -> str | None:
    """Compare live parser output `got` to recorded expected.<IMPL>.

    Returns None if matches, else a one-line drift description.
    """
    dyn = case["expected"]["dynamo"]
    kind, payload = classify_recorded(case, dyn)

    got_err = got.get("error")
    got_canonical = normalize_output(got)

    if kind == "missing":
        return (
            f"MISSING: no `expected.{IMPL}` block recorded; "
            f"parser produced {'error' if got_err else 'output'}"
        )

    assert payload is not None  # kind != "missing" guarantees this

    if kind == "unavailable":
        if got_err and got_err.startswith("UNAVAILABLE:"):
            return None
        return (
            f"DRIFT: expected unavailable ({payload['reason']!r}), parser now returns "
            + (f"error={got_err!r}" if got_err else f"output={got_canonical}")
        )

    if kind == "error":
        sub = payload["error_substring"]
        if got_err and sub in got_err:
            return None
        return f"DRIFT: expected error matching {sub!r}, parser now returns " + (
            f"error={got_err!r}" if got_err else f"output={got_canonical}"
        )

    # kind in ("match", "concrete") — both compare normalized output
    expected = normalize_output(dyn) if kind == "match" else payload

    if got_err:
        return f"DRIFT: expected output {expected}, parser errored: {got_err!r}"
    if got_canonical != expected:
        return f"DRIFT: expected {expected}, got {got_canonical}"
    return None


# ----------------------------------------------------------------------
# Merge mode (writes to YAML)


def _case_sort_key(case_id: str) -> tuple[int, str]:
    """`PARSER.batch.8.a` → (8, 'a'); `PARSER.batch.5` → (5, '')."""
    parts = case_id.split(".")
    return (int(parts[2]), parts[3] if len(parts) > 3 else "")


def _yaml_str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Multi-line strings render as `|` literal block."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _emit_yaml(fp: Path, doc: dict[str, Any]) -> None:
    """Write fixture YAML with SPDX header + optional `|2+` decode note."""
    if "cases" in doc and isinstance(doc["cases"], dict):
        doc["cases"] = {
            cid: doc["cases"][cid] for cid in sorted(doc["cases"], key=_case_sort_key)
        }
    body = yaml.dump(doc, sort_keys=False, allow_unicode=True, width=120)
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
    )
    if "|2+" in body:
        header += (
            "#\n"
            '# `|2+` simply means a single newline ("\\n"); '
            "see tests/parity/README.md for the full decode.\n"
        )
    fp.write_text(header + "\n" + body, encoding="utf-8")


def _build_merged_block(got: dict, dyn: dict) -> dict | None:
    """Translate parser output into a YAML-ready expected.<impl> block.

    Returns None when the case should be skipped (parser errored with
    something other than UNAVAILABLE — user must hand-record an
    `{error: <substring>}` block; auto-generating one would bake a
    fragile substring into the fixture).
    """
    err = got.get("error")
    if err and err.startswith("UNAVAILABLE:"):
        return {"unavailable": err[len("UNAVAILABLE:") :].strip()}
    if err:
        return None
    n_got = normalize_output(got)
    n_dyn = normalize_output(dyn)
    if n_got == n_dyn:
        # Same Python object → PyYAML auto-anchors on dump.
        return dyn
    return n_got


def merge_into_fixtures(
    all_outputs: dict[str, dict],
    overwrite: bool,
) -> tuple[int, int, int]:
    """Walk fixture YAMLs, write parser outputs into `expected.<impl>`.

    Returns (n_written, n_skipped_existing, n_skipped_error).
    """
    n_written = n_skipped_existing = n_skipped_error = 0
    for fp in sorted(FIXTURES.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text(encoding="utf-8"))
        family = doc["family"]
        if FAMILY_FILTER and family not in FAMILY_FILTER:
            continue
        cases = doc.get("cases") or {}
        changed = False
        for case_id, case in cases.items():
            key = f"{family}/{case_id}"
            if is_na_stub(case):
                continue
            got = all_outputs.get(key)
            if got is None:
                continue
            existing = (
                case.get("expected") if isinstance(case.get("expected"), dict) else {}
            )
            dyn = existing.get("dynamo")
            if not isinstance(dyn, dict):
                print(
                    f"  SKIP {key}: no `expected.dynamo` to compare against",
                    file=sys.stderr,
                )
                continue
            if IMPL in existing and not overwrite:
                n_skipped_existing += 1
                continue
            block = _build_merged_block(got, dyn)
            if block is None:
                print(
                    f"  SKIP {key}: parser errored — record "
                    f"`{IMPL}: {{error: <substring>}}` by hand",
                    file=sys.stderr,
                )
                n_skipped_error += 1
                continue
            new_expected = {k: v for k, v in existing.items()}
            new_expected[IMPL] = block
            case["expected"] = new_expected
            changed = True
            n_written += 1
        if changed:
            yaml.add_representer(str, _yaml_str_presenter)
            _emit_yaml(fp, doc)
            print(f"  wrote {fp.relative_to(REPO_ROOT)}")
    return n_written, n_skipped_existing, n_skipped_error


# ----------------------------------------------------------------------
# Main loop


def main() -> int:
    all_outputs: dict[str, dict] = {}
    n_total = n_clean = n_err = n_python_exc = n_skipped_na = 0
    drift: list[tuple[str, str]] = []

    for fp in sorted(FIXTURES.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        family = doc["family"]
        if FAMILY_FILTER and family not in FAMILY_FILTER:
            continue
        for case_id, case in doc["cases"].items():
            key = f"{family}/{case_id}"
            if is_na_stub(case):
                n_skipped_na += 1
                continue
            n_total += 1
            got = run_parser(family, case)
            all_outputs[key] = got
            if got["error"]:
                if got["error"].startswith("PYTHON_EXC:"):
                    n_python_exc += 1
                else:
                    n_err += 1
            else:
                n_clean += 1
            if not MERGE_MODE:
                d = check_drift(case, family, case_id, got)
                if d is not None:
                    drift.append((key, d))

    if FAMILY_FILTER and n_total == 0:
        print(
            f"ERROR: --family matched no cases: {', '.join(sorted(FAMILY_FILTER))}",
            file=sys.stderr,
        )
        return 2

    mode = "MERGE" if MERGE_MODE else "CHECK"
    print(
        f"IMPL={IMPL} mode={mode} total={n_total} clean={n_clean} "
        f"wrapper_error={n_err} python_exc={n_python_exc} skipped_na={n_skipped_na}"
    )

    if _args.dump:
        dump_fp = Path(_args.dump)
        dump_fp.write_text(json.dumps(all_outputs, indent=2, sort_keys=True))
        print(f"wrote {dump_fp}")

    if MERGE_MODE:
        n_written, n_skipped_existing, n_skipped_error = merge_into_fixtures(
            all_outputs, OVERWRITE
        )
        print(
            f"\n{n_written} written, {n_skipped_existing} already had "
            f"expected.{IMPL}, {n_skipped_error} skipped (parser error — "
            f"record `{{error: ...}}` by hand)."
        )
        if n_skipped_existing and not OVERWRITE:
            print("Pass --overwrite-if-exists to refresh those cases.")
        return 0

    # Default: drift check
    print(f"drifted: {len(drift)} / {n_total}")
    for key, d in drift[:40]:
        print(f"  {key}: {d}")
    if len(drift) > 40:
        print(f"  ... ({len(drift) - 40} more)")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())

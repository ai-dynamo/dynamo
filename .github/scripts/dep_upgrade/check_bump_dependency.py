#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight regression check for bump_dependency.py.

Runs the TRT-LLM bump against a temp copy of the real inputs with a synthetic
version and asserts it rewrites exactly MAIN_TOT.trtllm in
docs/fern/components/releases.data.ts — one line, inside the MAIN_TOT block,
no RELEASES pin touched, and the result still parses through
gen_llms_tables.py. Run from anywhere; exit 0 == safe to bump.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BUMP = REPO / ".github/scripts/dep_upgrade/bump_dependency.py"
GEN = REPO / "docs/fern/scripts/gen_llms_tables.py"
DATA_REL = "docs/fern/components/releases.data.ts"
# Real files apply() iterates for the trtllm framework; copy them so the run
# reaches the releases.data.ts target instead of dying on an earlier one.
SEED_FILES = ["container/context.yaml", "pyproject.toml", DATA_REL]

SYNTH = "9.9.9rc99"  # synthetic upstream TRT-LLM version, absent from real data


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> int:
    bump = load("bump_dependency", BUMP)
    gen = load("gen_llms_tables", GEN)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel in SEED_FILES:
            src = REPO / rel
            if not src.exists():
                fail(f"seed file missing in repo: {rel}")
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        data_path = root / DATA_REL
        before = data_path.read_text()

        # Invoke the bump logic with a synthetic TRT-LLM version.
        try:
            bump.apply("trtllm", SYNTH, root)
        except SystemExit as exc:
            fail(f"bump apply() raised SystemExit: {exc}")
        except Exception as exc:  # noqa: BLE001 - any crash is a red result
            fail(f"bump apply() raised {type(exc).__name__}: {exc}")

        after = data_path.read_text()
        if after == before:
            fail(f"{DATA_REL} was not modified by the bump")

        # The diff must touch exactly one line, and it must be MAIN_TOT.trtllm.
        b_lines, a_lines = before.splitlines(), after.splitlines()
        if len(b_lines) != len(a_lines):
            fail("line count changed; expected a single in-place pin edit")
        changed = [i for i, (x, y) in enumerate(zip(b_lines, a_lines)) if x != y]
        if len(changed) != 1:
            fail(f"expected exactly 1 changed line, got {len(changed)}: {changed}")
        idx = changed[0]
        new_line = a_lines[idx].strip()
        if new_line != f'trtllm: "{SYNTH}",':
            fail(f"changed line is not the MAIN_TOT trtllm pin: {new_line!r}")

        # The changed line must sit inside the MAIN_TOT literal, not a RELEASES row.
        tot_start = next(
            (i for i, ln in enumerate(a_lines) if "const MAIN_TOT" in ln), None
        )
        if tot_start is None:
            fail("MAIN_TOT declaration not found in bumped file")
        tot_end = next(
            (i for i in range(tot_start, len(a_lines)) if a_lines[i].strip() == "};"),
            None,
        )
        if not (tot_start < idx < tot_end):
            fail(
                f"changed line {idx} is outside the MAIN_TOT block "
                f"({tot_start}..{tot_end})"
            )

        # Synthetic version must appear exactly once (no accidental spillover).
        if after.count(SYNTH) != 1:
            fail(f"{SYNTH} appears {after.count(SYNTH)} times; expected exactly 1")

        # Per-release pins must be untouched: parse and cross-check.
        env = gen.parse_data_module(data_path)
        if env["MAIN_TOT"].get("trtllm") != SYNTH:
            fail(
                f"parsed MAIN_TOT.trtllm != {SYNTH}: {env['MAIN_TOT'].get('trtllm')!r}"
            )
        rel_pins = [
            r.get("pins", {}).get("trtllm")
            for r in env["RELEASES"]
            if r.get("pins", {}).get("trtllm")
        ]
        if SYNTH in rel_pins:
            fail("a RELEASES per-release trtllm pin was overwritten with the ToT value")
        if not rel_pins:
            fail("no RELEASES trtllm pins parsed; cross-check is meaningless")

    print("PASS: bump rewrote only MAIN_TOT.trtllm; RELEASES pins intact; file parses")
    print(f"  MAIN_TOT.trtllm -> {SYNTH}")
    print(f"  {len(rel_pins)} RELEASES trtllm pins left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())

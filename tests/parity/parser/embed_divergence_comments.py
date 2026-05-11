# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embed each registered cross-impl divergence as a `TODO(research)` comment
inside its sub-case fixture file.

For each (impl, family, case_id) in `KNOWN_DIVERGENCES` that maps to a
per-sub-case fixture (`PARSER.<mode>.<n>.yaml`), run the impl on the
fixture's `model_text`, capture its actual output, and insert a YAML comment
block right after the case's `expected:` data:

    PARSER.batch.8.b:
      ...
      expected: {...}     # Dynamo's output (the oracle)
      # TODO(research): vllm diverges — drops trailing normal_text...
      # vllm produces:
      #   calls: [...]
      #   normal_text: None

This makes divergences visible at the fixture level — no need to run the
harness to see what each impl actually does on the same input. Each comment
is a `TODO(research)` so a future investigator can decide whether Dynamo
should switch to the diverging behavior.

Run after `regenerate_fixtures.py` (which strips YAML comments on rewrite):

    PYTHONPATH=lib/bindings/python/src python3 -m tests.parity.parser.regenerate_fixtures --overwrite-if-exists
    PYTHONPATH=lib/bindings/python/src python3 -m tests.parity.parser.embed_divergence_comments

Idempotent: cases that already carry a matching `# TODO(research): <impl>`
comment are skipped.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tests.parity.parser import test_parity_parser  # noqa: E402
from tests.parity.parser import vllm as vllm_wrapper  # noqa: E402

FIXTURES = ROOT / "tests/parity/parser/fixtures"


def render_comment_block(impl: str, reason: str, got) -> list[str]:
    """Format an impl's output as YAML comment lines for embedding."""
    lines = [f"    # TODO(research): {impl} diverges — {reason}"]
    lines.append(f"    # {impl} produces:")
    if got.error:
        lines.append(f"    #   error: {got.error[:200]}")
        return lines
    lines.append(f"    #   calls: {got.calls}")
    lines.append(f"    #   normal_text: {got.normal_text!r}")
    return lines


def insert_comment(text: str, case_id: str, comment_lines: list[str]) -> str:
    """Insert comment_lines after the `expected:` block of `case_id`.

    The expected block ends at: the next case header (`  PARSER.batch.<n>(.<x>)?:` at 2-space indent),
    or end of file.
    """
    case_anchor = f"  {case_id}:"
    start = text.find(case_anchor)
    if start == -1:
        return text
    # Find next case header (same 2-space indent + PARSER.) after this one
    next_case_re = re.compile(r"^  PARSER\.[^\n]+:$", re.MULTILINE)
    rest = text[start + len(case_anchor) :]
    m = next_case_re.search(rest)
    if m:
        end_of_block = start + len(case_anchor) + m.start()
    else:
        end_of_block = len(text)
    # Insert comment lines just before next case (or EOF)
    insertion = "\n".join(comment_lines) + "\n"
    return text[:end_of_block] + insertion + text[end_of_block:]


def main() -> int:
    n_embedded = 0
    n_skipped = 0
    targets: dict[Path, list[tuple[str, str, str]]] = {}

    for (impl, family, case_id), reason in test_parity_parser.KNOWN_DIVERGENCES.items():
        if impl != "vllm":
            continue  # only embed vllm divergences for now (sglang n/a in container)
        # Only sub-case files (4+ dot parts in case_id)
        if len(case_id.split(".")) < 4:
            continue
        # Locate the per-sub-case fixture file
        parts = case_id.split(".")
        file_stem = ".".join(parts[:3])
        fp = FIXTURES / family / f"{file_stem}.yaml"
        if not fp.exists():
            n_skipped += 1
            continue
        targets.setdefault(fp, []).append((impl, family, case_id))

    for fp, items in sorted(targets.items()):
        text = fp.read_text()
        doc = yaml.safe_load(text)
        cases = doc["cases"]
        # Process cases in REVERSE order so insertions don't shift earlier indices
        for impl, family, case_id in sorted(items, key=lambda t: t[2], reverse=True):
            if case_id not in cases:
                continue
            case = cases[case_id]
            reason = test_parity_parser.KNOWN_DIVERGENCES[(impl, family, case_id)]
            # Run vLLM
            got = vllm_wrapper.parse(family, case["model_text"], case.get("tools"))
            # Skip if vLLM somehow matches now
            from tests.parity import common

            expected = common.ParseResult(
                calls=case["expected"]["calls"],
                normal_text=case["expected"].get("normal_text"),
            )
            if not got.error and common.canonical(got.to_dict()) == common.canonical(
                expected.to_dict()
            ):
                n_skipped += 1
                continue
            # Skip if comment already present
            if (
                f"# TODO(research): {impl} diverges" in text
                and case_id in text.split(f"  {case_id}:")[0:1][-1]
                if False
                else False
            ):
                # naive — recompute below
                pass
            # Better: check whether the case section already has a TODO comment
            anchor = f"  {case_id}:"
            section_start = text.find(anchor)
            next_case_re = re.compile(r"^  PARSER\.[^\n]+:$", re.MULTILINE)
            mm = next_case_re.search(text[section_start + len(anchor) :])
            section_end = section_start + len(anchor) + mm.start() if mm else len(text)
            section = text[section_start:section_end]
            if f"# TODO(research): {impl} diverges" in section:
                n_skipped += 1
                continue
            comment_lines = render_comment_block(impl, reason, got)
            text = insert_comment(text, case_id, comment_lines)
            n_embedded += 1
            print(f"  embedded {impl}/{family}/{case_id}")
        fp.write_text(text)

    print(f"\n{n_embedded} divergences embedded, {n_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

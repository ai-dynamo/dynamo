# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the POC's real frontend grammars without a model or GPU.

Run with a Python environment that contains a current XGrammar package:
    python3 lib/llm/tests/full_output_grammar_cpu.py
An isolated installation can be supplied with --xgrammar-path.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgrammar-path", type=Path)
    args = parser.parse_args()
    if args.xgrammar_path:
        sys.path.insert(0, str(args.xgrammar_path))
    import xgrammar as xgr

    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "dynamo-llm",
            "--no-default-features",
            "--lib",
            "full_output_poc_real_frontend_requests_and_parsing",
            "--",
            "--nocapture",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    cases = [
        json.loads(line.split("FULL_OUTPUT_POC_CASE=", 1)[1])
        for line in result.stdout.splitlines()
        if "FULL_OUTPUT_POC_CASE=" in line
    ]
    assert cases, "The Rust frontend test did not emit grammar cases."

    # A small vocabulary exercises token masks, including native marker tokens.
    # This is a test tokenizer, not a model tokenizer.
    special = [b"<think>", b"</think>", b"<tool_call>", b"</tool_call>", b"<eos>"]
    vocab = [bytes([i]) for i in range(256)] + special
    eos = len(vocab) - 1
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, stop_token_ids=[eos]))
    assertions = 0

    for case in cases:
        grammar = compiler.compile_structural_tag(case["structural_tag"])

        def accepts(text):
            matcher = xgr.GrammarMatcher(grammar)
            data = text.encode()
            while data:
                token = next(
                    (
                        256 + i
                        for i, marker in enumerate(special[:-1])
                        if data.startswith(marker)
                    ),
                    data[0],
                )
                if not matcher.accept_token(token):
                    return False
                data = data[len(vocab[token]) :]
            return matcher.accept_token(eos)

        output = case["valid_output"]
        assert accepts(output), f"Valid output rejected: {case}"
        assertions += 1
        if case["thinking"]:
            assert not accepts(output.split("</think>")[0]), case
            assert accepts(output.replace("private", "")), case
            assertions += 2
        if case["kind"] in (
            "schema",
            "named",
            "required",
            "named_json",
            "required_json",
        ):
            bad = output.replace('"answer":17', '"answer":"bad"')
            assert bad != output and not accepts(bad), case
            assertions += 1
        if case["kind"] in ("named", "required"):
            assert not accepts(output.replace('"name": "answer"', '"name": "other"')), (
                case
            )
            assertions += 1
        if case["kind"] == "schema":
            assert not accepts(
                output.replace('"answer":17', '"answer":17,"extra":1')
            ), case
            assertions += 1
            if case["thinking"] and not case["prefilled"]:
                assert not accepts('{"answer":17}'), case
                assertions += 1
            if not case["thinking"]:
                assert not accepts('<think>private</think>{"answer":17}'), case
                assertions += 1
    print(
        f"Passed: {len(cases)} real frontend grammars, {assertions} token-matcher assertions."
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract pinned N-2 adapter evidence without importing historical Dynamo code.

The excerpts are executed only by the explicit backend contract tests, against
the release's real SamplingParams. They are not a replacement for mixed-release
HTTP, discovery, preprocessing, or disaggregated tests.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from ..common.paths import ROOT

RELEASES = (
    ("1.4.0", "03014943323e78feb5bd672ef08b72caea0918ac"),
    ("1.5.0", "b83b1d9304ebfc624709ac46db32b1b6f1ff1615"),
)
FIXTURE_DIRECTORY = Path("components/src/dynamo/vllm/tests/fixtures/protocol_releases")
HANDLER = "components/src/dynamo/vllm/handlers.py"
LOGPROBS = "components/src/dynamo/common/backend/logprobs.py"
PREPROCESSOR = "lib/llm/src/preprocessor.rs"
CONTEXT = "container/context.yaml"


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def excerpt(source: str, name: str) -> dict:
    """Keep exact lines (including comments), rejecting ambiguous declarations."""
    candidates = []
    for node in ast.parse(source).body:
        if (isinstance(node, ast.FunctionDef) and node.name == name) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            candidates.append(node)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one declaration of {name}")
    node = candidates[0]
    if isinstance(node, ast.FunctionDef) and node.decorator_list:
        raise ValueError(f"Decorated declaration requires explicit review: {name}")
    text = "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    return {"source": text, "sha256": sha256(text), "line": node.lineno}


def writer_evidence(source: str) -> dict:
    """Extract the release's literal passthrough list, not an inferred schema."""
    start = source.index("    fn sampling_passthrough_args<")
    end = source.index("    fn backend_extra_args<", start)
    text = source[start:end]
    matches = re.findall(r"for key in (\[[^\]]+\])\s*\{", text)
    if len(matches) != 1:
        raise ValueError("Historical writer no longer has one literal key list")
    keys = ast.literal_eval(matches[0])
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
        raise ValueError("Historical writer key list is not a string array")
    return {
        "source": text,
        "sha256": sha256(text),
        "line": source[:start].count("\n") + 1,
        "keys": keys,
    }


def generate(repo: Path, release: str, commit: str) -> str:
    def read(path: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"], text=True
        )

    sources = {path: read(path) for path in (HANDLER, LOGPROBS, PREPROCESSOR, CONTEXT)}
    context = yaml.safe_load(sources[CONTEXT])["vllm"]["cuda13.0"]
    functions = {
        "build_sampling_params": excerpt(sources[HANDLER], "build_sampling_params"),
        "parse_logprob_options": excerpt(sources[LOGPROBS], "parse_logprob_options"),
        "_parse_non_negative_int": excerpt(
            sources[LOGPROBS], "_parse_non_negative_int"
        ),
    }
    constants = {}
    if release == "1.5.0":
        for name in (
            "_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY",
            "_ROUTER_HINT_EXTRA_ARGS_KEY",
        ):
            constants[name] = excerpt(sources[HANDLER], name)
    result = {
        "_generated": "Do not edit; run python -m scripts.protocol_compatibility generate-release-fixtures",
        "_license": "Apache-2.0; Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.",
        "dynamo_release": release,
        "dynamo_commit": commit,
        "engine_image": f"{context['runtime_image']}:{context['runtime_image_tag']}",
        "engine_version": context["runtime_image_tag"].removeprefix("v").split("-")[0],
        "source_sha256": {path: sha256(source) for path, source in sources.items()},
        "functions": functions,
        "constants": constants,
        "writer": writer_evidence(sources[PREPROCESSOR]),
    }
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    for release, commit in RELEASES:
        target = args.repo / FIXTURE_DIRECTORY / f"dynamo-{release}.json"
        expected = generate(args.repo, release, commit)
        if args.check:
            if not target.exists() or target.read_text() != expected:
                raise SystemExit(f"Stale or missing release fixture: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected)
    return 0

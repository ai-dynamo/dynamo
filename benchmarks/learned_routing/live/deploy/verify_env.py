# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the live Dynamo venv inside the vLLM image before any model loads.

Checks that the image's vLLM and torch are the pinned ones, that ``dynamo._core`` is the built
wheel's binding (hash) and ``dynamo`` imports from the staged source, that the frontend and vLLM
worker modules import, and that every ``engine_client.generate(...)`` call site in
``dynamo.vllm.handlers`` passes only keywords the installed ``AsyncLLM.generate`` accepts (the
compat shim drops ``session_id``, which vLLM 0.24 lacks). Writes a JSON report; exit 1 on failure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_call_keywords(handlers: Path) -> list[tuple[int, list[str]]]:
    calls = []
    for node in ast.walk(ast.parse(handlers.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "engine_client"
        ):
            calls.append(
                (node.lineno, [k.arg for k in node.keywords if k.arg is not None])
            )
    return calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--wheel-manifest", type=Path, required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict = {"python": sys.version, "executable": sys.executable, "problems": []}
    problems = report["problems"]

    import torch
    import vllm

    report["vllm"] = vllm.__version__
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    if vllm.__version__ != args.vllm_version:
        problems.append(f"vllm {vllm.__version__} != {args.vllm_version}")

    import dynamo
    import dynamo._core
    import dynamo.frontend.main  # noqa: F401
    import dynamo.vllm.handlers as handlers
    import dynamo.vllm.main  # noqa: F401

    core = Path(dynamo._core.__file__).resolve()
    wheel = json.loads(args.wheel_manifest.read_text())
    report["core_so"] = str(core)
    report["core_so_sha256"] = sha256(core)
    report["wheel_core_so_sha256"] = wheel.get("core_so_sha256")
    report["ai_dynamo_runtime"] = importlib.metadata.version("ai-dynamo-runtime")
    report["dynamo_paths"] = [str(Path(p).resolve()) for p in dynamo.__path__]
    if not str(core).startswith(str(args.venv.resolve())):
        problems.append(f"dynamo._core loads from {core}, not the venv")
    if report["core_so_sha256"] != wheel.get("core_so_sha256"):
        problems.append("dynamo._core differs from the built wheel's binding")
    src_dynamo = str((args.src / "components" / "src" / "dynamo").resolve())
    if src_dynamo not in report["dynamo_paths"]:
        problems.append(f"dynamo package does not include {src_dynamo}")

    from vllm.v1.engine.async_llm import AsyncLLM

    accepted = inspect.signature(AsyncLLM.generate).parameters
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    calls = generate_call_keywords(Path(handlers.__file__))
    report["generate_call_sites"] = len(calls)
    for line, keywords in calls:
        unknown = [k for k in keywords if k not in accepted and not var_kw]
        if unknown:
            problems.append(f"handlers.py:{line} passes {unknown} to AsyncLLM.generate")
    session_gate = getattr(handlers, "_engine_generate_session_support", None)
    report["session_id_shim"] = session_gate is not None
    if "session_id" not in accepted and session_gate is None:
        problems.append(
            "vLLM lacks generate(session_id=...) and the compat shim is missing"
        )

    report["ok"] = not problems
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"ok": report["ok"], "problems": problems}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

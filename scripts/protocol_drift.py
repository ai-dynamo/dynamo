# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract and compare versioned serving contracts without importing upstream code.

This is a declaration/implementation-change detector, not a conformance oracle.
The JSON inventory can feed admission inventories and reference generation; every
changed executable body needs review or runtime evidence before claiming parity.
Requires only Python's standard library and a local Git object database.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol_dependencies import DependencyResolver
from protocol_review_report import render_review

FORMAT_VERSION = 1
VLLM_ENDPOINTS = ("/v1/chat/completions", "/v1/completions")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def commit(repo: Path, revision: str) -> str:
    # Require immutable input, and avoid Git option/revision-expression injection.
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")
    resolved = git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}").strip()
    if resolved != revision:
        raise ValueError("revision must identify a commit, not an annotated tag object")
    return resolved


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tool_provenance(*names: str) -> dict[str, Any]:
    """Identify the actual extractor tools, including uncommitted helper changes.

    Names are fixed by each entrypoint, not discovered from arbitrary untracked
    files. Repository-relative labels keep private checkout paths out of reports.
    Python is recorded because AST serialization is interpreter-version dependent.
    """
    scripts = Path(__file__).resolve().parent
    return {
        "python_version": platform.python_version(),
        "tools_sha256": {
            f"scripts/{name}": hashlib.sha256((scripts / name).read_bytes()).hexdigest()
            for name in sorted(names)
        },
    }


class SemanticAST(ast.NodeTransformer):
    """Ignore documentation and formatting, but preserve executable semantics."""

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        # Pydantic field prose is not a wire contract. Preserve aliases, bounds,
        # discriminator, default/default_factory and all other keyword arguments.
        if isinstance(node.func, ast.Name) and node.func.id == "Field":
            node.keywords = [
                kw for kw in node.keywords if kw.arg not in {"description", "title"}
            ]
        return node


def expression(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def module_contract(source: str) -> dict[str, Any]:
    """Retain declarations and hash behavior independently of source formatting.

    Inheritance is recorded, not evaluated. References to external classes/types
    remain references; importing arbitrary engine code would not be safe here.
    """
    tree = SemanticAST().visit(ast.parse(source))
    classes: dict[str, Any] = {}
    functions: dict[str, str] = {}
    bindings: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = {}
            methods = {}
            other = []
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name
                ):
                    fields[child.target.id] = {
                        "annotation": expression(child.annotation),
                        "default": expression(child.value),
                    }
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[child.name] = digest(ast.dump(child))
                else:
                    other.append(ast.dump(child))
            classes[node.name] = {
                "bases": [expression(base) for base in node.bases],
                "decorators": [expression(item) for item in node.decorator_list],
                "keywords": [ast.dump(item) for item in node.keywords],
                "fields": fields,
                "methods": methods,
                "other": other,
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = digest(ast.dump(node))
        else:
            # Includes imports, aliases, constants, conditionals, and route setup.
            bindings.append(ast.dump(node))
    return {"classes": classes, "functions": functions, "bindings": bindings}


def vllm_source(path: str) -> bool:
    """Support both monolithic and split upstream OpenAI server layouts."""
    if path in {"vllm/sampling_params.py", "vllm/entrypoints/chat_utils.py"}:
        return True
    if path.endswith(".py") and path.startswith(
        (
            "vllm/entrypoints/serve/engine/",
            "vllm/entrypoints/serve/exception_handling/",
            "vllm/entrypoints/generate/base/",
            "vllm/renderers/",
        )
    ):
        return True
    prefix = "vllm/entrypoints/openai/"
    if not path.startswith(prefix) or not path.endswith(".py"):
        return False
    relative = path[len(prefix) :]
    return relative.startswith(
        ("chat_completion/", "completion/", "engine/")
    ) or relative in {
        "protocol.py",
        "serving_chat.py",
        "serving_completion.py",
        "serving_engine.py",
        "api_server.py",
    }


def snapshot(repo: Path, revision: str, target: str) -> dict[str, Any]:
    revision = commit(repo, revision)
    if target != "vllm":
        raise ValueError(f"no source extractor registered for {target!r}")
    all_paths = git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
    paths = sorted(path for path in all_paths if vllm_source(path))
    sources = {}

    def read_source(path: str) -> str:
        if path not in sources:
            sources[path] = git(repo, "show", f"{revision}:{path}")
        return sources[path]

    coverage = DependencyResolver(all_paths, read_source).collect(paths)
    paths = sorted(set(paths) | coverage["module_consumers"].keys())
    modules = {}
    for path in paths:
        source = read_source(path)
        modules[path] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "contract": module_contract(source),
        }
    names = {
        name for module in modules.values() for name in module["contract"]["classes"]
    }
    required = {
        "ChatCompletionRequest",
        "CompletionRequest",
        "ChatCompletionResponse",
        "CompletionResponse",
        "SamplingParams",
    }
    if not required <= names:
        raise ValueError(
            f"incomplete upstream extraction; missing {sorted(required - names)}"
        )
    return {
        "format_version": FORMAT_VERSION,
        "target": target,
        "upstream_commit": revision,
        "endpoints": list(VLLM_ENDPOINTS),
        "modules": modules,
        "dependency_coverage": coverage,
        "limitations": [
            "Static declarations and source changes do not establish runtime behavior.",
            "Repository-local type dependencies are followed statically, not evaluated as JSON Schema.",
            "Unresolved/dynamic/external dependencies are listed explicitly; primitive library implementations remain outside scope.",
            "Engine/tokenizer/model behavior needs runtime probes.",
        ],
    }


def changes(
    before: Any, after: Any, path: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Produce stable leaf changes, preserving missing versus explicit null."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            location = (*path, key)
            if key not in before:
                result.append(
                    {"path": list(location), "kind": "added", "after": after[key]}
                )
            elif key not in after:
                result.append(
                    {"path": list(location), "kind": "removed", "before": before[key]}
                )
            else:
                result.extend(changes(before[key], after[key], location))
        return result
    if before == after:
        return []
    return [{"path": list(path), "kind": "changed", "before": before, "after": after}]


def compare(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    if before["target"] != after["target"]:
        raise ValueError("cannot compare different target servers")
    # Source digests provide provenance only: whitespace/comments must not create drift.
    old = {path: item["contract"] for path, item in before["modules"].items()}
    new = {path: item["contract"] for path, item in after["modules"].items()}
    result = changes(old, new)
    for item in result:
        item["id"] = digest(item)
        item["status"] = "Unverified"
        item[
            "next_step"
        ] = "Review upstream semantics and run affected conformance probes."
        item["runtime_evidence"] = []
        consumers = set()
        for inventory in (before, after):
            coverage = inventory.get("dependency_coverage", {})
            path = item["path"]
            if len(path) >= 3 and path[1] in {"classes", "functions"}:
                consumers.update(
                    coverage.get("symbol_consumers", {}).get(f"{path[0]}:{path[2]}", [])
                )
            else:
                consumers.update(coverage.get("module_consumers", {}).get(path[0], []))
        item["reachable_consumers"] = sorted(consumers)
    return result


def build_report(
    before: dict[str, Any], after: dict[str, Any], dynamo_sha: str, extractor_sha: str
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "target": before["target"],
        "endpoints": before["endpoints"],
        "previous_upstream_commit": before["upstream_commit"],
        "candidate_upstream_commit": after["upstream_commit"],
        "dynamo_commit": dynamo_sha,
        "extractor_sha256": extractor_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owner": "frontend",
        "baseline_inventory_sha256": digest(before),
        "candidate_inventory_sha256": digest(after),
        "changes": compare(before, after),
        "limitations": after["limitations"],
        "dependency_coverage": {
            "before": before.get("dependency_coverage"),
            "after": after.get("dependency_coverage"),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["vllm"], default="vllm")
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--previous", required=True, help="full upstream commit SHA")
    parser.add_argument("--candidate", required=True, help="full upstream commit SHA")
    parser.add_argument(
        "--dynamo-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="fail if either source snapshot has unresolved type dependencies",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; preserve previous reports")
    before = snapshot(args.upstream_repo, args.previous, args.target)
    after = snapshot(args.upstream_repo, args.candidate, args.target)
    dynamo_sha = git(args.dynamo_repo, "rev-parse", "HEAD").strip()
    report = build_report(
        before,
        after,
        dynamo_sha,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    report["dynamo_tracked_diff_sha256"] = hashlib.sha256(
        git(args.dynamo_repo, "diff", "--binary", "HEAD", "--").encode()
    ).hexdigest()
    report["provenance"] = tool_provenance(
        "protocol_drift.py", "protocol_dependencies.py", "protocol_review_report.py"
    )
    report["require_complete_coverage"] = args.require_complete_coverage
    write_json(args.output_dir / "previous.json", before)
    write_json(args.output_dir / "candidate.json", after)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "summary.md").write_text(
        render_review(
            [("report.json", report)], "drift" if args.fail_on_drift else "none"
        )
    )
    print(
        f"{len(report['changes'])} review candidates; report: {args.output_dir / 'report.json'}"
    )
    print(f"Developer review: {args.output_dir / 'summary.md'}")
    incomplete = not all(
        value["complete"] for value in report["dependency_coverage"].values()
    )
    return (
        1
        if (args.fail_on_drift and report["changes"])
        or (args.require_complete_coverage and incomplete)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

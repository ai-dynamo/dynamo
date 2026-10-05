# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract Python declarations, selected source bodies and reachable dependencies."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from ..common.provenance import digest
from ..common.source import commit, git
from .dependencies import DependencyResolver

FORMAT_VERSION = 1
VLLM_ENDPOINTS = ("/v1/chat/completions", "/v1/completions")


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


def snapshot(
    repo: Path, revision: str, target: str, *, require_protocol_roots: bool = True
) -> dict[str, Any]:
    """Read selected source and dependencies.

    Vocabulary generation requires all five roots. Direct assessment disables
    this precondition and reports missing request roots as coverage findings;
    response roots are outside its initial comparison scope.
    """
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

    coverage = DependencyResolver(all_paths, read_source).collect(
        paths, require_roots=require_protocol_roots
    )
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
    if require_protocol_roots and not required <= names:
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

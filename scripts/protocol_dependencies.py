# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, source-only type dependency closure for native endpoint declarations.

No imports are executed. Repository-local imports, re-exports, aliases, forward
references, nested models and bases are followed. External packages and dynamic
definitions are reported as unresolved, never silently treated as complete.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable
from typing import Any

ROOT_CLASSES = frozenset(
    {
        "ChatCompletionRequest",
        "CompletionRequest",
        "ChatCompletionResponse",
        "CompletionResponse",
        "SamplingParams",
    }
)
BUILTIN_TYPES = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "object",
        "type",
        "None",
        "Ellipsis",
    }
)
# These are declaration primitives, not vendored implementations. Their runtime
# behavior remains outside the static coverage claim and is listed in the report.
LEAF_MODULES = frozenset({"typing", "typing_extensions", "builtins", "collections.abc"})
PYDANTIC_LEAVES = frozenset(
    {"BaseModel", "Field", "ConfigDict", "AliasChoices", "AliasPath"}
)


def module_name(path: str) -> str:
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


class DependencyResolver:
    """Resolve reachable declarations within one immutable upstream Git tree."""

    def __init__(self, paths: list[str], read_source: Callable[[str], str]):
        self.paths = {module_name(path): path for path in paths if path.endswith(".py")}
        self.read_source = read_source
        self.indexes: dict[str, dict[str, ast.AST]] = {}
        self.imports: dict[str, dict[str, str]] = {}
        self.conditional: dict[str, set[str]] = {}
        self.reachable: dict[str, set[str]] = defaultdict(set)
        self.symbol_consumers: dict[str, set[str]] = defaultdict(set)
        self.unresolved: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.leaves: set[str] = set()
        self.class_bindings: dict[str, dict[str, str | None]] = {}

    def problem(self, source: str, symbol: str, reason: str, root: str) -> None:
        self.unresolved[(source, symbol, reason)].add(root)

    def index(self, module: str) -> None:
        if module in self.indexes:
            return
        path = self.paths[module]
        self.indexes[module] = {}
        self.imports[module] = {}
        self.conditional[module] = set()
        tree = ast.parse(self.read_source(path))
        package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]

        def statements(nodes: list[ast.stmt], conditional: bool = False) -> None:
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        self.imports[module][name] = (
                            alias.name if alias.asname else name
                        )
                        if conditional:
                            self.conditional[module].add(name)
                elif isinstance(node, ast.ImportFrom):
                    prefix = node.module or ""
                    if node.level:
                        parents = package.split(".")
                        prefix = ".".join(
                            parents[: len(parents) - node.level + 1]
                            + ([prefix] if prefix else [])
                        )
                    for alias in node.names:
                        name = alias.asname or alias.name
                        self.imports[module][name] = f"{prefix}.{alias.name}"
                        if conditional:
                            self.conditional[module].add(name)
                elif isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    self.indexes[module][node.name] = node
                    if conditional:
                        self.conditional[module].add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.indexes[module][target.id] = node
                            if conditional:
                                self.conditional[module].add(target.id)
                elif type(node).__name__ == "TypeAlias":  # Python 3.12+ syntax
                    self.indexes[module][node.name.id] = node
                elif isinstance(node, ast.If):
                    # TYPE_CHECKING declarations are statically usable for type
                    # discovery. Other conditions are unresolved runtime choices.
                    type_checking = ast.unparse(node.test) in {
                        "TYPE_CHECKING",
                        "typing.TYPE_CHECKING",
                    }
                    statements(node.body, conditional or not type_checking)
                    statements(node.orelse, True)
                elif isinstance(node, ast.Try):
                    statements(node.body, True)
                    for handler in node.handlers:
                        statements(handler.body, True)
                    statements(node.orelse, True)
                    statements(node.finalbody, True)

        statements(tree.body)

    def canonical(self, module: str, symbol: str) -> str:
        head, dot, tail = symbol.partition(".")
        prefix = self.imports[module].get(head, head)
        return prefix + (dot + tail if dot else "")

    def resolve(
        self, module: str, symbol: str, root: str, seen: set[tuple[str, str]]
    ) -> tuple[str | None, str] | None:
        resolved = self._resolve(module, symbol, root, seen)
        if resolved:
            self.class_bindings[f"{self.paths[module]}:{symbol}"] = {
                "source": resolved[0],
                "name": resolved[1],
            }
        return resolved

    def _resolve(
        self, module: str, symbol: str, root: str, seen: set[tuple[str, str]]
    ) -> tuple[str | None, str] | None:
        self.index(module)
        path = self.paths[module]
        if (module, symbol) in seen:
            if not isinstance(self.indexes[module].get(symbol), ast.ClassDef):
                self.problem(path, symbol, "cyclic alias or re-export", root)
                return None
            return path, symbol  # Recursive models are valid.
        seen = seen | {(module, symbol)}
        self.reachable[path].add(root)
        self.symbol_consumers[f"{path}:{symbol}"].add(root)
        head = symbol.split(".")[0]
        if head in self.conditional[module]:
            self.problem(path, symbol, "conditional definition/import", root)
        node = self.indexes[module].get(symbol)
        if node is not None:
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    self.expression(module, base, root, seen)
                for field in node.body:
                    if isinstance(field, ast.AnnAssign):
                        self.expression(module, field.annotation, root, seen)
                        # Defaults are captured in the module contract. Resolve
                        # named constants/factories too, but don't execute them.
                        if field.value is not None:
                            self.expression(
                                module, field.value, root, seen, strings=False
                            )
                return path, symbol
            elif (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                or type(node).__name__ == "TypeAlias"
            ):
                if node.value is not None:
                    if isinstance(node.value, (ast.Name, ast.Attribute)):
                        return self.resolve(module, ast.unparse(node.value), root, seen)
                    if isinstance(node.value, ast.Call):
                        self.problem(
                            path,
                            symbol,
                            "call-created alias/value requires runtime evaluation",
                            root,
                        )
                    self.expression(module, node.value, root, seen)
            else:
                self.problem(
                    path,
                    symbol,
                    "callable implementation is not recursively resolved",
                    root,
                )
            return
        qualified = self.canonical(module, symbol)
        if (
            qualified in BUILTIN_TYPES
            or ("." in qualified and qualified.rpartition(".")[0] in LEAF_MODULES)
            or qualified in {f"pydantic.{name}" for name in PYDANTIC_LEAVES}
        ):
            self.leaves.add(qualified)
            if qualified == "pydantic.BaseModel":
                return None, "BaseModel"
            return
        # Support common base/typing names in synthetic or old declarations
        # lacking imports only for BaseModel; other missing names are explicit.
        if qualified == "BaseModel":
            self.leaves.add("pydantic.BaseModel")
            return None, "BaseModel"
        parts = qualified.split(".")
        for split in range(len(parts) - 1, 0, -1):
            imported_module = ".".join(parts[:split])
            if imported_module in self.paths:
                return self.resolve(
                    imported_module, ".".join(parts[split:]), root, seen
                )
        reason = (
            "external dependency source unavailable"
            if head in self.imports[module]
            else "unresolved name or dynamic/star import"
        )
        self.problem(path, qualified, reason, root)

    def expression(
        self,
        module: str,
        node: ast.AST,
        root: str,
        seen: set[tuple[str, str]],
        strings: bool = True,
    ) -> None:
        if isinstance(node, (ast.Name, ast.Attribute)):
            self.resolve(module, ast.unparse(node), root, seen)
        elif isinstance(node, ast.Constant):
            if strings and isinstance(node.value, str):
                try:
                    forward = ast.parse(node.value, mode="eval").body
                except SyntaxError:
                    self.problem(
                        self.paths[module],
                        node.value,
                        "unparseable forward reference",
                        root,
                    )
                else:
                    # A quoted string expression cannot resolve into a type.
                    if isinstance(forward, ast.Constant):
                        self.problem(
                            self.paths[module],
                            node.value,
                            "unresolved forward reference",
                            root,
                        )
                    else:
                        self.expression(module, forward, root, seen)
        elif isinstance(node, ast.Subscript):
            self.expression(module, node.value, root, seen)
            name = self.canonical(module, ast.unparse(node.value)).rpartition(".")[2]
            if name == "Literal":
                return  # String literal choices are values, not forward types.
            if name == "Annotated":
                args = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                self.expression(module, args[0], root, seen)
                for metadata in args[1:]:
                    self.expression(module, metadata, root, seen, strings=False)
            else:
                self.expression(module, node.slice, root, seen)
        elif isinstance(node, ast.Call):
            self.expression(module, node.func, root, seen, strings=False)
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                self.expression(module, argument, root, seen, strings=False)
        else:
            for child in ast.iter_child_nodes(node):
                self.expression(module, child, root, seen, strings=strings)

    def collect(self, selected: list[str]) -> dict[str, Any]:
        roots = []
        for path in selected:
            module = module_name(path)
            self.index(module)
            for name in sorted(ROOT_CLASSES & self.indexes[module].keys()):
                node = self.indexes[module][name]
                root = f"{path}:{name}"
                roots.append(root)
                self.resolve(module, name, root, set())
                if isinstance(node, ast.ClassDef):
                    # Field-specific reachability helps reviewers connect an
                    # imported alias/model change back to its public consumers.
                    for field in node.body:
                        if isinstance(field, ast.AnnAssign) and isinstance(
                            field.target, ast.Name
                        ):
                            self.expression(
                                module,
                                field.annotation,
                                f"{root}.{field.target.id}",
                                set(),
                            )
        missing = ROOT_CLASSES - {root.rpartition(":")[2] for root in roots}
        if missing:
            raise ValueError(
                f"incomplete upstream extraction; missing {sorted(missing)}"
            )
        diagnostics = [
            {
                "source": source,
                "symbol": symbol,
                "reason": reason,
                "roots": sorted(consumers),
            }
            for (source, symbol, reason), consumers in sorted(self.unresolved.items())
        ]
        return {
            "complete": not diagnostics,
            "roots": roots,
            "module_consumers": {
                path: sorted(consumers)
                for path, consumers in sorted(self.reachable.items())
            },
            "unresolved": diagnostics,
            "primitive_leaves": sorted(self.leaves),
            "class_bindings": dict(sorted(self.class_bindings.items())),
            "symbol_consumers": {
                symbol: sorted(consumers)
                for symbol, consumers in sorted(self.symbol_consumers.items())
            },
            "scope": "Repository-local static type closure; primitive library implementations and runtime behavior are excluded.",
        }

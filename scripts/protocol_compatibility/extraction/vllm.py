# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize pinned vLLM request declarations without executing engine imports.

This intentionally does not evaluate validators, factories or external packages.
Unknown facts are preserved as coverage failures, not replaced with permissive
types. Referenced repository-local models and aliases are expanded structurally.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from scripts.protocol_compatibility.common.contracts import (
    Contract,
    CoverageDiagnostic,
    EndpointContract,
    FieldContract,
    canonical_union,
)
from scripts.protocol_compatibility.common.source import commit, git
from scripts.protocol_compatibility.extraction.dependencies import (
    DependencyResolver,
    module_name,
)
from scripts.protocol_compatibility.extraction.python_source import snapshot

from .request_fields import REQUESTS, request_fields

PRIMITIVES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "None": "null",
    "NoneType": "null",
    "Any": "any",
    "object": "any",
}
CONSTRAINTS = frozenset(
    {
        "gt",
        "ge",
        "lt",
        "le",
        "min_length",
        "max_length",
        "pattern",
        "multiple_of",
        "strict",
        "allow_inf_nan",
        "max_digits",
        "decimal_places",
    }
)


class UnknownFact(ValueError):
    """An unsupported source construct, not a tooling or Git failure."""


def literal(node: ast.AST) -> Any:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise UnknownFact(f"requires evaluation: {ast.unparse(node)}") from error
    if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
        raise UnknownFact(f"not a JSON literal: {ast.unparse(node)}")
    return value


class NativeContractExtractor:
    def __init__(self, paths: list[str], read_source, inventory: dict[str, Any]):
        self.resolver = DependencyResolver(paths, read_source)
        self.inventory = inventory
        self.diagnostics: dict[tuple[str, str, str], set[str]] = {}

    def problem(self, endpoint: str, path: str, aspect: str, reason: str) -> None:
        self.diagnostics.setdefault((endpoint, path, aspect), set()).add(reason)

    def constant(self, module: str | None, node: ast.AST, seen=()) -> Any:
        """Bounded constant folding, never eval/import/call upstream code."""
        if len(seen) > 32:
            raise UnknownFact("constant dependency depth exceeds 32")
        if isinstance(node, (ast.Name, ast.Attribute)) and module:
            symbol = ast.unparse(node)
            if (module, symbol) in seen:
                raise UnknownFact(f"cyclic constant: {module}.{symbol}")
            origin, value = self.definition(module, symbol)
            if (
                not isinstance(value, (ast.Assign, ast.AnnAssign))
                or value.value is None
            ):
                raise UnknownFact(f"not a static constant: {symbol}")
            return self.constant(origin, value.value, (*seen, (module, symbol)))
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.LShift: operator.lshift,
            ast.Pow: operator.pow,
        }
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left = self.constant(
                module, node.left, (*seen, ("expression", str(id(node))))
            )
            right = self.constant(
                module, node.right, (*seen, ("expression", str(id(node))))
            )
            if (
                type(left) is not int
                or type(right) is not int
                or abs(left) > 2**128
                or abs(right) > 2**128
            ):
                raise UnknownFact("constant arithmetic requires bounded integers")
            if isinstance(node.op, (ast.Pow, ast.LShift)) and not 0 <= right <= 128:
                raise UnknownFact("constant exponent/shift exceeds bounds")
            result = operations[type(node.op)](left, right)
            if result.bit_length() > 1024:
                raise UnknownFact("constant arithmetic result exceeds 1024 bits")
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.constant(
                module, node.operand, (*seen, ("expression", str(id(node))))
            )
            if type(value) not in (int, float):
                raise UnknownFact("non-numeric constant unary operator")
            return -value if isinstance(node.op, ast.USub) else value
        return literal(node)

    def model_config(self, module: str, node: ast.ClassDef, seen=()) -> dict[str, Any]:
        key = (module, node.name)
        if key in seen:
            raise UnknownFact("cyclic model configuration inheritance")
        config = {}
        for base in node.bases:
            name = self.resolver.canonical(module, ast.unparse(base))
            if name in {
                "pydantic.BaseModel",
                "BaseModel",
                "typing.TypedDict",
                "typing_extensions.TypedDict",
            }:
                continue
            origin, parent = self.definition(module, ast.unparse(base))
            if not isinstance(parent, ast.ClassDef):
                raise UnknownFact("unresolved model configuration inheritance")
            config.update(self.model_config(origin, parent, (*seen, key)))
        for child in node.body:
            if isinstance(child, ast.ClassDef) and child.name == "Config":
                raise UnknownFact(
                    "legacy Config class requires configuration interpretation"
                )
            targets = (
                child.targets
                if isinstance(child, ast.Assign)
                else [child.target]
                if isinstance(child, ast.AnnAssign)
                else []
            )
            if not any(
                isinstance(target, ast.Name) and target.id == "model_config"
                for target in targets
            ):
                continue
            value = child.value
            if isinstance(value, ast.Dict):
                if any(key is None for key in value.keys):
                    raise UnknownFact("expanded model configuration")
                options = [
                    (literal(key), val) for key, val in zip(value.keys, value.values)
                ]
            elif (
                isinstance(value, ast.Call)
                and self.resolver.canonical(module, ast.unparse(value.func))
                in {"ConfigDict", "pydantic.ConfigDict"}
                and not value.args
            ):
                options = [(kw.arg, kw.value) for kw in value.keywords]
            else:
                raise UnknownFact("dynamic model configuration")
            for name, value in options:
                if name not in {
                    "populate_by_name",
                    "validate_by_name",
                    "validate_by_alias",
                    "extra",
                    "strict",
                    "protected_namespaces",
                    "arbitrary_types_allowed",
                    "title",
                    "frozen",
                }:
                    raise UnknownFact(f"unresolved model configuration: {name}")
                config[name] = self.constant(module, value)
        if node.keywords:
            raise UnknownFact("model class options require interpretation")
        for name in (
            "populate_by_name",
            "validate_by_name",
            "validate_by_alias",
            "strict",
        ):
            if name in config and type(config[name]) is not bool:
                raise UnknownFact(f"non-boolean model configuration: {name}")
        return config

    def definition(
        self, module: str, symbol: str, seen: tuple[tuple[str, str], ...] = ()
    ) -> tuple[str, ast.AST]:
        if (module, symbol) in seen:
            raise UnknownFact(f"cyclic import/alias: {module}.{symbol}")
        self.resolver.index(module)
        if symbol in self.resolver.conditional[module]:
            raise UnknownFact(f"conditional definition: {module}.{symbol}")
        node = self.resolver.indexes[module].get(symbol)
        if node is not None:
            return module, node
        qualified = self.resolver.canonical(module, symbol)
        pieces = qualified.split(".")
        for index in range(len(pieces) - 1, 0, -1):
            imported = ".".join(pieces[:index])
            if imported in self.resolver.paths:
                return self.definition(
                    imported, ".".join(pieces[index:]), (*seen, (module, symbol))
                )
        raise UnknownFact(f"unresolved dependency (no pinned source): {qualified}")

    def wire_type(
        self, module: str, node: ast.AST, stack: tuple[tuple[str, str], ...] = ()
    ) -> dict[str, Any]:
        self.resolver.index(module)
        if isinstance(node, ast.Constant):
            if node.value is None:
                return {"type": "null"}
            if isinstance(node.value, str):
                parsed = ast.parse(node.value, mode="eval").body
                if isinstance(parsed, ast.Constant):
                    raise UnknownFact("unresolved quoted type")
                return self.wire_type(module, parsed, stack)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return canonical_union(
                [
                    self.wire_type(module, part, stack)
                    for part in (node.left, node.right)
                ]
            )
        if isinstance(node, ast.Subscript):
            name = self.resolver.canonical(module, ast.unparse(node.value)).split(".")[
                -1
            ]
            args = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            if name == "Literal":
                return {"enum": sorted([literal(arg) for arg in args], key=repr)}
            if name == "Annotated":
                # Metadata may modify the wire type; do not silently erase it.
                if any(
                    not isinstance(arg, ast.Call)
                    or ast.unparse(arg.func).split(".")[-1] != "Field"
                    for arg in args[1:]
                ):
                    raise UnknownFact("unresolved Annotated metadata")
                result = self.wire_type(module, args[0], stack)
                constraints = {}
                for metadata in args[1:]:
                    keywords, values = self.field_keywords(metadata, module)
                    if (
                        set(keywords)
                        & {"alias", "validation_alias", "default", "default_factory"}
                        or metadata.args
                    ):
                        raise UnknownFact(
                            "Annotated alias/default requires field-level interpretation"
                        )
                    constraints.update(values)
                return {
                    **result,
                    **({"constraints": constraints} if constraints else {}),
                }
            if name in {"Union", "Optional"}:
                members = [self.wire_type(module, arg, stack) for arg in args]
                if name == "Optional":
                    members.append({"type": "null"})
                return canonical_union(members)
            if name in {"list", "List", "Sequence", "Iterable"} and len(args) == 1:
                return {
                    "type": "array",
                    "items": self.wire_type(module, args[0], stack),
                }
            if name in {"dict", "Dict", "Mapping"} and len(args) == 2:
                key = self.wire_type(module, args[0], stack)
                if key != {"type": "string"}:
                    raise UnknownFact(
                        "non-string JSON object keys require normalization"
                    )
                return {
                    "type": "object",
                    "additional_properties": self.wire_type(module, args[1], stack),
                }
            raise UnknownFact(f"unresolved generic type: {ast.unparse(node)}")
        if isinstance(node, (ast.Name, ast.Attribute)):
            symbol = ast.unparse(node)
            qualified = self.resolver.canonical(module, symbol)
            # Only known language/typing primitives, not same-named user classes.
            tail = qualified.split(".")[-1]
            if tail in PRIMITIVES and (
                qualified == tail or qualified.startswith(("typing.", "builtins."))
            ):
                if tail not in self.resolver.indexes[module]:
                    return {"type": PRIMITIVES[tail]}
            if (
                qualified in {"dict", "builtins.dict", "list", "builtins.list"}
                and tail not in self.resolver.indexes[module]
            ):
                return (
                    {"type": "object", "additional_properties": {"type": "any"}}
                    if tail == "dict"
                    else {"type": "array", "items": {"type": "any"}}
                )
            origin, definition = self.definition(module, symbol)
            key = (origin, symbol)
            if key in stack:
                raise UnknownFact(
                    f"recursive type requires schema reference: {origin}.{symbol}"
                )
            stack = (*stack, key)
            if (
                isinstance(definition, (ast.Assign, ast.AnnAssign))
                or type(definition).__name__ == "TypeAlias"
            ):
                return self.wire_type(origin, definition.value, stack)
            if isinstance(definition, ast.ClassDef):
                return self.model_type(origin, definition, stack)
        raise UnknownFact(f"unresolved type expression: {ast.unparse(node)}")

    def model_type(
        self,
        module: str,
        node: ast.ClassDef,
        stack: tuple[tuple[str, str], ...],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.model_config(module, node) if config is None else config
        properties = {}
        for base in node.bases:
            name = self.resolver.canonical(module, ast.unparse(base))
            if name in {
                "pydantic.BaseModel",
                "BaseModel",
                "typing.TypedDict",
                "typing_extensions.TypedDict",
            }:
                continue
            origin, parent = self.definition(module, ast.unparse(base))
            if not isinstance(parent, ast.ClassDef) or (origin, parent.name) in stack:
                raise UnknownFact("unresolved model inheritance")
            properties.update(
                self.model_type(
                    origin, parent, (*stack, (origin, parent.name)), config
                )["properties"]
            )
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                name = child.target.id
                if (
                    name == "model_config"
                    or name.startswith("_")
                    or ast.unparse(child.annotation).startswith("ClassVar[")
                ):
                    continue
                value_type = self.wire_type(module, child.annotation, stack)
                names, required, default, constraints = self.field_facts(
                    name, child.value, module=module, config=config
                )
                constraints = {**value_type.pop("constraints", {}), **constraints}
                for wire_name in names:
                    properties[wire_name] = {
                        "wire_type": value_type,
                        "required": required,
                        "default": default,
                        "constraints": constraints,
                    }
        if node.keywords:
            raise UnknownFact("model class options require interpretation")
        return {"type": "object", "properties": properties}

    def field_keywords(
        self, node: ast.Call, module: str | None = None
    ) -> tuple[dict[str, ast.AST], dict[str, Any]]:
        if ast.unparse(node.func).split(".")[-1] != "Field":
            raise UnknownFact("dynamic field default")
        keywords = {}
        constraints = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise UnknownFact("expanded Field keyword arguments")
            keywords[keyword.arg] = keyword.value
            if keyword.arg in CONSTRAINTS:
                constraints[keyword.arg] = self.constant(module, keyword.value)
            elif keyword.arg not in {
                "alias",
                "validation_alias",
                "serialization_alias",
                "default",
                "default_factory",
                "description",
                "title",
                "deprecated",
                "examples",
                "json_schema_extra",
                "repr",
                "exclude",
            }:
                raise UnknownFact(f"unresolved Field option: {keyword.arg}")
        return keywords, constraints

    def input_names(
        self, name: str, default: ast.AST | None, config: dict[str, Any]
    ) -> list[str]:
        if (
            not isinstance(default, ast.Call)
            or ast.unparse(default.func).split(".")[-1] != "Field"
        ):
            return [name]
        keywords = {kw.arg: kw.value for kw in default.keywords}
        if None in keywords:
            raise UnknownFact("expanded Field keywords may alter aliases")
        alias = keywords.get("validation_alias")
        if alias is None or (isinstance(alias, ast.Constant) and alias.value is None):
            alias = keywords.get("alias")
        if alias is None or (isinstance(alias, ast.Constant) and alias.value is None):
            return [name]
        if (
            isinstance(alias, ast.Call)
            and ast.unparse(alias.func).split(".")[-1] == "AliasChoices"
            and not alias.keywords
        ):
            names = [literal(arg) for arg in alias.args]
        else:
            names = [literal(alias)]
        if not names or not all(isinstance(item, str) for item in names):
            raise UnknownFact("non-string or empty input alias")
        by_alias = config.get("validate_by_alias", True)
        by_name = config.get(
            "validate_by_name", config.get("populate_by_name", not by_alias)
        )
        if not by_alias:
            names = []
        if by_name:
            names.append(name)
        if not names:
            raise UnknownFact("both alias and field-name validation disabled")
        return sorted(set(names))

    def field_facts(
        self,
        name: str,
        default: ast.AST | None,
        *,
        module: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[list[str], bool, dict[str, Any], dict[str, Any]]:
        config = config or {}
        if module:
            self.resolver.index(module)
        names, constraints = self.input_names(name, default, config), {}
        if "strict" in config:
            constraints["strict"] = config["strict"]
        if isinstance(default, ast.Call):
            keywords, declared_constraints = self.field_keywords(default, module)
            constraints.update(declared_constraints)
            if "default_factory" in keywords:
                factory = ast.unparse(keywords["default_factory"])
                if module and factory in self.resolver.indexes[module]:
                    raise UnknownFact(f"shadowed default factory: {factory}")
                if factory in {"list", "dict", "str", "bool", "int", "float"}:
                    values = {
                        "list": [],
                        "dict": {},
                        "str": "",
                        "bool": False,
                        "int": 0,
                        "float": 0.0,
                    }
                    return (
                        names,
                        False,
                        {"kind": "value", "value": values[factory]},
                        constraints,
                    )
                raise UnknownFact(f"unresolved default factory: {factory}")
            default = keywords.get("default", default.args[0] if default.args else None)
        if default is None or (
            isinstance(default, ast.Constant) and default.value is Ellipsis
        ):
            return names, True, {"kind": "absent"}, constraints
        return (
            names,
            False,
            {"kind": "value", "value": self.constant(module, default)},
            constraints,
        )

    def extract(self, revision: str) -> Contract:
        endpoints = {}
        for endpoint, request in REQUESTS.items():
            result = EndpointContract()
            endpoints[endpoint] = result
            try:
                declarations = request_fields(
                    self.inventory, request, tolerate_unresolved_aliases=True
                )
            except ValueError as error:
                self.problem(endpoint, "*", "inheritance", str(error))
                continue
            result.fields_complete = True
            config = {}
            try:
                roots = [
                    module_name(path)
                    for path, value in self.inventory["modules"].items()
                    if request in value["contract"]["classes"]
                ]
                if len(roots) != 1:
                    raise UnknownFact("ambiguous request configuration")
                origin, root = self.definition(roots[0], request)
                if not isinstance(root, ast.ClassDef):
                    raise UnknownFact("request root is not a class")
                config = self.model_config(origin, root)
            except UnknownFact as error:
                result.fields_complete = False
                self.problem(endpoint, "*", "configuration", str(error))
            for name, declaration in sorted(declarations.items()):
                path = declaration["source"]
                item = FieldContract(
                    name,
                    [name],
                    source=[
                        {
                            "path": path,
                            "symbol": declaration["declared_in"],
                            "target": "vllm",
                        }
                    ],
                    references=[declaration["annotation"]],
                )
                result.fields[name] = item
                try:
                    item.wire_type = self.wire_type(
                        module_name(path),
                        ast.parse(declaration["annotation"], mode="eval").body,
                    )
                    members = item.wire_type.get("any_of", [item.wire_type])
                    item.nullable = any(
                        member.get("type") in {"null", "any"}
                        or None in member.get("enum", [])
                        for member in members
                    )
                except UnknownFact as error:
                    self.problem(endpoint, name, "type", str(error))
                try:
                    default = (
                        ast.parse(declaration["default"], mode="eval").body
                        if declaration["default"] is not None
                        else None
                    )
                    (
                        item.wire_names,
                        item.required,
                        item.default,
                        item.constraints,
                    ) = self.field_facts(
                        name, default, module=module_name(path), config=config
                    )
                    if item.wire_type is not None:
                        item.constraints = {
                            **item.wire_type.pop("constraints", {}),
                            **item.constraints,
                        }
                except UnknownFact as error:
                    self.problem(endpoint, name, "declaration", str(error))
                    try:
                        item.wire_names = self.input_names(name, default, config)
                    except UnknownFact as alias_error:
                        item.wire_names = []
                        self.problem(endpoint, name, "input_names", str(alias_error))
                if not result.fields_complete:
                    item.wire_names = []
        return Contract(
            "vllm",
            revision,
            endpoints,
            [
                CoverageDiagnostic(endpoint, path, aspect, "; ".join(sorted(reasons)))
                for (endpoint, path, aspect), reasons in sorted(
                    self.diagnostics.items()
                )
            ],
            {
                "extraction": "source-only; validators/factories are not executed",
                "external_dependencies": "unavailable unless source is explicitly pinned; unresolved facts are diagnostics",
            },
        )


def extract_native(repo: Path, revision: str) -> tuple[Contract, dict[str, Any]]:
    revision = commit(repo, revision)
    inventory = snapshot(repo, revision, "vllm", require_protocol_roots=False)
    paths = git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
    extractor = NativeContractExtractor(
        paths, lambda path: git(repo, "show", f"{revision}:{path}"), inventory
    )
    return extractor.extract(revision), inventory

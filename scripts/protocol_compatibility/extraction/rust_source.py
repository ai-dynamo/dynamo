# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tree-sitter syntax reader with bounded Serde interpretation, not a compiler.

Rust syntax and declaration boundaries come from the pinned Rust grammar.
Error-recovered trees are rejected. Token-list helpers below interpret only the
supported attribute/type subset; they do not discover Rust declarations.
No build scripts, macros, or dependencies execute.
Registry sources are read from archives whose checksum matches Cargo.lock.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_rust
from tree_sitter import Language, Node, Parser

from scripts.protocol_compatibility.common.provenance import digest
from scripts.protocol_compatibility.common.source import git


class RustUnknown(ValueError):
    """The bounded source reader cannot establish a requested contract fact."""


def parse(source: str) -> Node:
    root = (
        Parser(Language(tree_sitter_rust.language())).parse(source.encode()).root_node
    )
    if root.has_error:
        raise RustUnknown(
            "Rust syntax error or unsupported syntax in tree-sitter grammar"
        )
    return root


def node_tokens(node: Node) -> list[str]:
    """Project parsed syntax into the contract interpreter's token vocabulary."""
    if node.type in {"line_comment", "block_comment"}:
        return []
    if node.type == "raw_string_literal":
        content = next(
            child for child in node.children if child.type == "string_content"
        )
        return [json.dumps(content.text.decode())]
    if node.type in {"string_literal", "char_literal"} or not node.children:
        return [node.text.decode()]
    return [token for child in node.children for token in node_tokens(child)]


def tokens(source: str) -> list[str]:
    # A token-tree wrapper accepts type/attribute fragments without guessing their
    # Rust syntactic context. Full source files always go through parse() directly.
    root = parse("protocol_tokens! {" + source + "\n}")
    invocation = root.named_children[0]
    tree = next(child for child in invocation.children if child.type == "token_tree")
    return node_tokens(tree)[1:-1]


def group(items: list[str], start: int) -> tuple[list[str], int]:
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    if start >= len(items) or items[start] not in pairs:
        raise RustUnknown("expected delimited Rust source")
    opening = items[start]
    depth = 1
    for index in range(start + 1, len(items)):
        if items[index] == opening:
            depth += 1
        elif items[index] == pairs[opening]:
            depth -= 1
            if depth == 0:
                return items[start + 1 : index], index + 1
    raise RustUnknown(f"unclosed Rust delimiter {opening}")


def split(items: list[str], separator: str = ",") -> list[list[str]]:
    results, current, stack = [], [], []
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for token in items:
        if token == separator and not stack:
            if current:
                results.append(current)
            current = []
            continue
        current.append(token)
        if token in pairs:
            stack.append(pairs[token])
        elif stack and token == stack[-1]:
            stack.pop()
    if current:
        results.append(current)
    return results


def attributes(items: list[str], start: int = 0) -> tuple[list[list[str]], int]:
    result = []
    while start < len(items) and items[start] == "#":
        offset = start + 2 if items[start + 1] == "!" else start + 1
        body, start = group(items, offset)
        result.append(body)
    return result, start


def serde_options(attrs: list[list[str]]) -> dict[str, list[list[str]]]:
    options: dict[str, list[list[str]]] = {}
    for attr in attrs:
        if attr and attr[0] == "serde":
            if len(attr) < 3 or attr[1] != "(":
                raise RustUnknown("unrecognized serde attribute")
            body, _ = group(attr, 1)
            for option in split(body):
                options.setdefault(option[0], []).append(option[1:])
    return options


def string_option(options: dict[str, list[list[str]]], name: str) -> list[str]:
    values = []
    for option in options.get(name, []):
        if not option and name == "default":
            continue
        if len(option) != 2 or option[0] != "=" or not option[1].startswith('"'):
            raise RustUnknown(f"unresolved serde {name} option")
        values.append(json.loads(option[1]))
    return values


@dataclass
class RustItem:
    name: str
    kind: str
    source: str
    crate: str
    attrs: list[list[str]]
    header: list[str]
    body: list[str]
    owner: str = ""

    def evidence(self) -> dict:
        return {
            "path": self.source,
            "symbol": self.name,
            "target": "dynamo",
            "semantic_sha256": digest([self.header, self.body]),
        }


@dataclass
class RustSources:
    items: list[RustItem] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    dependencies: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    imports: dict[str, dict[str, str]] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=dict)

    def add(self, path: str, source: str, crate: str = "dynamo_llm") -> None:
        root = parse(source)  # Reject the entire file before mutating the index.
        self.sources[path] = source
        self.modules[path] = self.module_path(path, crate)
        self.imports[path] = {}
        self._items(root, path, crate)

    @staticmethod
    def module_path(path: str, crate: str) -> str:
        relative = path.split("/src/", 1)[-1].removesuffix(".rs")
        parts = relative.split("/")
        if parts[-1] in {"lib", "mod"}:
            parts.pop()
        return "::".join([crate, *parts])

    def add_import(
        self, tree: list[str], path: str, prefix: list[str] | None = None
    ) -> None:
        prefix = prefix or []
        if "{" in tree:
            start = tree.index("{")
            inner, _ = group(tree, start)
            for child in split(inner):
                self.add_import(child, path, [*prefix, *tree[:start]])
            return
        combined = [*prefix, *tree]
        if "as" in combined:
            index = combined.index("as")
            alias, combined = combined[index + 1], combined[:index]
        else:
            alias = combined[-1]
        target = "".join(combined)
        module = self.modules[path].split("::")
        if target.startswith("crate::"):
            target = module[0] + target[len("crate") :]
        elif target.startswith("self::"):
            target = "::".join(module) + target[len("self") :]
        elif target.startswith("super::"):
            while target.startswith("super::"):
                module.pop()
                target = target[len("super::") :]
            target = "::".join([*module, target])
        if target.endswith("::self"):
            target = target.removesuffix("::self")
            alias = target.split("::")[-1]
        if alias == "*":
            alias = "*" + target
        previous = self.imports[path].get(alias)
        self.imports[path][alias] = (
            "@ambiguous" if previous is not None and previous != target else target
        )

    def imported_name(self, name: str, source: str) -> str:
        head, separator, tail = name.partition("::")
        imported = self.imports.get(source, {}).get(head)
        return (imported + separator + tail) if imported else name

    def _items(self, root: Node, path: str, crate: str, owner: str = "") -> None:
        pending = []
        for node in root.named_children:
            if node.type in {"line_comment", "block_comment"}:
                continue
            if node.type == "attribute_item":
                pending.append(node_tokens(node)[2:-1])
                continue
            attrs, pending = pending, []
            if any(attr[:2] == ["cfg", "("] and "test" in attr for attr in attrs):
                continue
            if node.type == "use_declaration":
                self.add_import(node_tokens(node.child_by_field_name("argument")), path)
                continue
            kind = {"function_item": "fn"}.get(
                node.type, node.type.removesuffix("_item")
            )
            if kind not in {"struct", "enum", "type", "fn", "const", "impl", "mod"}:
                continue
            body_node = node.child_by_field_name("body")
            header = [
                token
                for child in node.children
                if child != body_node and child.type != ";"
                for token in node_tokens(child)
            ]
            if kind in {"impl", "mod"}:
                if body_node is not None:
                    self._items(body_node, path, crate, " ".join(header))
                continue
            name = node.child_by_field_name("name").text.decode().removeprefix("r#")
            body = node_tokens(body_node) if body_node is not None else []
            if body and body[0] == "{":
                body = body[1:-1]
            self.items.append(
                RustItem(name, kind, path, crate, attrs, header, body, owner)
            )

    def resolve(
        self, name: str, source: str, seen: tuple[tuple[str, str], ...] = ()
    ) -> RustItem:
        if (name, source) in seen:
            raise RustUnknown(f"cyclic Rust import/re-export: {name}")
        seen = (*seen, (name, source))
        local = [
            item
            for item in self.items
            if item.name == name
            and item.source == source
            and item.kind in {"struct", "enum", "type"}
        ]
        if len(local) == 1:
            return local[0]
        imported = self.imported_name(name, source)
        if imported != name:
            return self.resolve(imported, source, seen)
        if "::" in name:
            qualified = name
            if qualified.startswith("crate::") and source in self.modules:
                qualified = (
                    self.modules[source].split("::")[0] + qualified[len("crate") :]
                )
            known_crates = {module.split("::")[0] for module in self.modules.values()}
            if qualified.split("::")[0] not in known_crates and source in self.modules:
                relative = self.modules[source] + "::" + qualified
                if any(
                    module == relative.rpartition("::")[0]
                    for module in self.modules.values()
                ):
                    qualified = relative
            exact = [
                item
                for item in self.items
                if self.modules[item.source] + "::" + item.name == qualified
                and item.kind in {"struct", "enum", "type"}
            ]
            if len(exact) == 1:
                return exact[0]
            namespace, _, symbol = qualified.rpartition("::")
            results = []
            for module_source, module in self.modules.items():
                if module == namespace:
                    try:
                        results.append(self.resolve(symbol, module_source, seen))
                    except RustUnknown:
                        continue  # Try alternate re-export paths; ambiguity is checked below.
            unique = {(item.source, item.name): item for item in results}
            if len(unique) == 1:
                return next(iter(unique.values()))
        else:
            results = []
            for alias, target in self.imports.get(source, {}).items():
                if alias.startswith("*"):
                    try:
                        results.append(
                            self.resolve(target.removesuffix("*") + name, source, seen)
                        )
                    except RustUnknown:
                        continue
            unique = {(item.source, item.name): item for item in results}
            if len(unique) == 1:
                return next(iter(unique.values()))
            if len(unique) > 1:
                raise RustUnknown(f"ambiguous Rust glob import: {name}")
        if source or "::" in name:
            raise RustUnknown(f"unresolved Rust import/type: {name} from {source}")
        base = name.split("::")[-1]
        matches = [
            item
            for item in self.items
            if item.name == base and item.kind in {"struct", "enum", "type"}
        ]
        # Qualified dependency names must never fall back to a similarly named
        # struct from another crate when the pinned source is unavailable.
        prefix = name.split("::")[0]
        if prefix in {"dynamo_protocols", "async_openai", "dynamo_llm"}:
            matches = [item for item in matches if item.crate == prefix]
        else:
            current_crates = {
                item.crate for item in self.items if item.source == source
            }
            same_crate = [item for item in matches if item.crate in current_crates]
            if same_crate:
                matches = same_crate
        local = [item for item in matches if item.source == source]
        if len(local) == 1:
            return local[0]
        if len(matches) == 1:
            return matches[0]
        raise RustUnknown(
            f"unresolved/ambiguous Rust type {name}: {len(matches)} definitions"
        )


def load_sources(
    repo: Path, revision: str, crate_cache: Path | None = None
) -> RustSources:
    result = RustSources()
    paths = git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
    for path in paths:
        if path.endswith(".rs") and (
            path.startswith("lib/llm/src/protocols/")
            or path
            in {
                "lib/llm/src/protocols.rs",
                "lib/llm/src/preprocessor.rs",
                "lib/llm/src/types.rs",
            }
        ):
            try:
                result.add(path, git(repo, "show", f"{revision}:{path}"))
            except RustUnknown as error:
                result.problems.append(f"{path}: {error}")
    lock = tomllib.loads(git(repo, "show", f"{revision}:Cargo.lock"))
    for package in lock["package"]:
        if package["name"] not in {"dynamo-protocols", "async-openai"}:
            continue
        identity = f'{package["name"]}-{package["version"]}'
        archives = list(crate_cache.rglob(identity + ".crate")) if crate_cache else []
        if not archives:
            result.problems.append(f"pinned crate source unavailable: {identity}")
            continue
        archive = archives[0]
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        if checksum != package.get("checksum"):
            raise ValueError(f"Cargo.lock checksum mismatch for {identity}")
        result.dependencies.append(
            {
                "package": package["name"],
                "version": package["version"],
                "sha256": checksum,
                "source": package["source"],
            }
        )
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                prefix = identity + "/src/"
                if (
                    member.isfile()
                    and member.name.startswith(prefix)
                    and member.name.endswith(".rs")
                ):
                    reader = bundle.extractfile(member)
                    if reader is None:
                        raise ValueError(f"missing archive member: {member.name}")
                    with reader:
                        source = reader.read().decode()
                    try:
                        result.add(
                            "crate:" + member.name,
                            source,
                            package["name"].replace("-", "_"),
                        )
                    except RustUnknown as error:
                        result.problems.append(f"{member.name}: {error}")
    return result

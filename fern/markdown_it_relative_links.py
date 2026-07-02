# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""markdown-it-py core plugin for repository-relative documentation links."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token


OPTION_KEY = "fern_relative_links"
RENDERERS: dict[str, Any] = {}


@dataclass
class LinkRewriteResult:
    replacements: list[dict[str, Any]] = field(default_factory=list)
    skipped_replacements: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LinkRewriteConfig:
    source_path: Path
    docs_root: Path
    repo_root: Path
    repository: str
    ref: str
    result: LinkRewriteResult


@dataclass
class LinkTarget:
    destination: str
    parser_kinds: set[str]
    block_start_line: int | None
    syntax: str
    tokens: list[tuple[Token, str]] = field(default_factory=list)
    reference: dict[str, Any] | None = None

    def replace(self, destination: str) -> None:
        for token, attribute in self.tokens:
            token.attrSet(attribute, destination)
        if self.reference is not None:
            self.reference["href"] = destination


def _is_relative_path(destination: str) -> bool:
    if destination.startswith(("#", "/", "\\")):
        return False
    parsed = urlsplit(destination)
    return not parsed.scheme and not parsed.netloc and bool(parsed.path)


def _resolved_path(source_path: Path, destination: str) -> Path | None:
    if not _is_relative_path(destination):
        return None
    path = unquote(urlsplit(destination).path)
    return (source_path.parent / path).resolve()


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _append_url_suffix(url: str, destination: str) -> str:
    parsed = urlsplit(destination)
    if parsed.query:
        url += f"?{parsed.query}"
    if parsed.fragment:
        url += f"#{parsed.fragment}"
    return url


def _github_url(path: Path, config: LinkRewriteConfig, target: LinkTarget) -> str:
    relative_path = quote(path.relative_to(config.repo_root).as_posix(), safe="/@:+")
    quoted_ref = quote(config.ref, safe="/@:+")
    if "image" in target.parser_kinds:
        url = (
            f"https://raw.githubusercontent.com/{config.repository}/"
            f"{quoted_ref}/{relative_path}"
        )
        return _append_url_suffix(url, target.destination)

    object_kind = "tree" if path.is_dir() else "blob"
    url = (
        f"https://github.com/{config.repository}/{object_kind}/"
        f"{quoted_ref}/{relative_path}"
    )
    return _append_url_suffix(url, target.destination)


def _issue(target: LinkTarget) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "destination": target.destination,
        "parserKinds": sorted(target.parser_kinds),
        "syntax": target.syntax,
    }
    if target.block_start_line is not None:
        issue["blockStartLine"] = target.block_start_line
    return issue


def _inline_targets(state: StateCore) -> tuple[list[LinkTarget], dict[str, LinkTarget]]:
    targets: list[LinkTarget] = []
    reference_targets: dict[str, LinkTarget] = {}
    for block_token in state.tokens:
        line = block_token.map[0] + 1 if block_token.map else None
        for token in block_token.children or []:
            if token.type == "link_open":
                attribute = "href"
                parser_kind = "link"
            elif token.type == "image":
                attribute = "src"
                parser_kind = "image"
            else:
                continue

            destination = token.attrGet(attribute)
            if destination is None:
                continue
            label = token.meta.get("label")
            if label is None:
                targets.append(
                    LinkTarget(
                        destination,
                        {parser_kind},
                        line,
                        "inline",
                        [(token, attribute)],
                    )
                )
                continue

            target = reference_targets.setdefault(
                label,
                LinkTarget(destination, set(), line, "reference"),
            )
            target.parser_kinds.add(parser_kind)
            target.tokens.append((token, attribute))
    return targets, reference_targets


def _collect_targets(state: StateCore) -> list[LinkTarget]:
    targets, reference_targets = _inline_targets(state)
    references = state.env.get("references", {})
    for label, target in reference_targets.items():
        reference = references[label]
        target.destination = reference["href"]
        target.reference = reference
        if reference.get("map"):
            target.block_start_line = reference["map"][0] + 1
        targets.append(target)
    return targets


def _rewrite_relative_links(state: StateCore) -> None:
    config = state.md.options.get(OPTION_KEY)
    if config is None:
        config = state.md.options.get("mdformat", {}).get(OPTION_KEY)
    if not isinstance(config, LinkRewriteConfig):
        return

    planned: list[tuple[LinkTarget, str, dict[str, Any]]] = []
    for target in _collect_targets(state):
        path = _resolved_path(config.source_path, target.destination)
        if path is None or is_within(path, config.docs_root):
            continue

        issue = _issue(target)
        if not is_within(path, config.repo_root):
            config.result.unresolved.append(
                {**issue, "reason": "destination resolves outside the repository"}
            )
            continue
        if not path.exists():
            config.result.unresolved.append(
                {**issue, "reason": "destination does not exist"}
            )
            continue

        replacement = _github_url(path, config, target)
        planned.append((target, replacement, {**issue, "replacement": replacement}))

    if config.result.unresolved:
        config.result.skipped_replacements.extend(report for _, _, report in planned)
        return

    for target, replacement, report in planned:
        target.replace(replacement)
        config.result.replacements.append(report)


def relative_link_plugin(md: MarkdownIt) -> None:
    """Rewrite external repository-relative links after inline parsing."""

    md.core.ruler.after("inline", "fern_relative_links", _rewrite_relative_links)


def update_mdit(md: MarkdownIt) -> None:
    """Expose the plugin through mdformat's parser-extension interface."""

    md.use(relative_link_plugin)

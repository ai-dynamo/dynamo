#!/usr/bin/env -S uv run --script

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "markdown-it-py==4.2.0",
# ]
# ///

"""Rewrite docs links without rendering or otherwise reformatting Markdown.

This spike uses markdown-it-py as a syntax oracle and a source scanner only to
locate candidate destination spans. Each scanner candidate is replaced with a
unique temporary URL before parsing. A candidate is eligible for mutation only
when that exact URL appears in the parser's link, image, or reference output.

The source tree is never modified. The input tree is copied to --output, where
relative links that escape the docs tree are rewritten to exact GitHub URLs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkDestination
from markdown_it.token import Token


MD_EXTENSIONS = {".md", ".mdx"}
SENTINEL_PREFIX = "https://link-rewrite-spike.invalid/"
REFERENCE_PREFIX = re.compile(r"(?m)^[ \t]{0,3}\[(?:\\.|[^\[\]\r\n])+\]:[ \t]*")
SENTINEL = re.compile(rf"^{re.escape(SENTINEL_PREFIX)}(?P<id>\d+)$")


@dataclass
class Candidate:
    id: int
    start: int
    end: int
    destination: str
    syntax: str
    parser_kinds: set[str] = field(default_factory=set)

    @property
    def sentinel(self) -> str:
        return f"{SENTINEL_PREFIX}{self.id}"


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    destination: str
    replacement: str
    parser_kinds: tuple[str, ...]


@dataclass(frozen=True)
class MdxTag:
    start: int
    end: int
    name: str
    indent: int
    closing: bool
    self_closing: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repository", default="ai-dynamo/dynamo")
    parser.add_argument("--ref", default="main", help="Git ref used in generated URLs")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output tree"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when links cannot be mapped or resolved",
    )
    return parser.parse_args()


def build_parser() -> MarkdownIt:
    return MarkdownIt(
        "commonmark",
        {
            "html": True,
            "inline_definitions": True,
            "store_labels": True,
        },
    ).enable(["strikethrough", "table"])


def destination_span(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Return the source span inside optional angle brackets."""

    if start >= end:
        return None
    if source[start] == "<" and source[end - 1] == ">":
        if start + 1 == end - 1:
            return None
        return start + 1, end - 1
    return start, end


def scan_inline_destinations(source: str) -> list[tuple[int, int, str, str]]:
    found: list[tuple[int, int, str, str]] = []
    for marker in re.finditer(r"\]\(", source):
        position = marker.end()
        while position < len(source) and source[position] in " \t\r\n":
            position += 1
        parsed = parseLinkDestination(source, position, len(source))
        if not parsed.ok:
            continue
        span = destination_span(source, position, parsed.pos)
        if span is not None:
            found.append((*span, parsed.str, "inline"))
    return found


def scan_reference_destinations(source: str) -> list[tuple[int, int, str, str]]:
    found: list[tuple[int, int, str, str]] = []
    for marker in REFERENCE_PREFIX.finditer(source):
        position = marker.end()
        # CommonMark permits a reference destination on the following line.
        if position < len(source) and source[position] in "\r\n":
            if source.startswith("\r\n", position):
                position += 2
            else:
                position += 1
            indent_start = position
            while position < len(source) and source[position] in " \t":
                position += 1
            if position - indent_start > 3:
                continue
        parsed = parseLinkDestination(source, position, len(source))
        if not parsed.ok:
            continue
        span = destination_span(source, position, parsed.pos)
        if span is not None:
            found.append((*span, parsed.str, "reference"))
    return found


def scan_candidates(source: str) -> list[Candidate]:
    scanned = scan_inline_destinations(source) + scan_reference_destinations(source)
    unique: dict[tuple[int, int], tuple[str, str]] = {}
    for start, end, destination, syntax in scanned:
        unique.setdefault((start, end), (destination, syntax))

    candidates: list[Candidate] = []
    previous_end = -1
    for candidate_id, ((start, end), (destination, syntax)) in enumerate(
        sorted(unique.items())
    ):
        if start < previous_end:
            raise ValueError(f"overlapping scanner candidates at offsets {start}:{end}")
        candidates.append(Candidate(candidate_id, start, end, destination, syntax))
        previous_end = end
    return candidates


def replace_candidate_spans(source: str, candidates: Iterable[Candidate]) -> str:
    rewritten = source
    for candidate in sorted(candidates, key=lambda item: item.start, reverse=True):
        rewritten = (
            rewritten[: candidate.start]
            + candidate.sentinel
            + rewritten[candidate.end :]
        )
    return rewritten


def walk_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    for token in tokens:
        yield token
        if token.children:
            yield from walk_tokens(token.children)


def mask_characters(source: str, start: int, end: int) -> str:
    return "".join(
        "\r" if char == "\r" else "\n" if char == "\n" else " "
        for char in source[start:end]
    )


def mdx_tag_end(source: str, start: int) -> int | None:
    """Find a top-level closing angle bracket for a JSX component tag."""

    quote_character: str | None = None
    brace_depth = 0
    position = start
    while position < len(source):
        character = source[position]
        if quote_character is not None:
            if character == "\\":
                position += 2
                continue
            if character == quote_character:
                quote_character = None
        elif character in {'"', "'"}:
            quote_character = character
        elif character == "{":
            brace_depth += 1
        elif character == "}" and brace_depth:
            brace_depth -= 1
        elif character == ">" and brace_depth == 0:
            return position + 1
        position += 1
    return None


def standalone_mdx_tags(source: str) -> dict[int, MdxTag]:
    tags: dict[int, MdxTag] = {}
    line_start = 0
    while line_start < len(source):
        content_start = line_start
        while content_start < len(source) and source[content_start] in " \t":
            content_start += 1
        if content_start < len(source) and source[content_start] == "<":
            name_start = content_start + 1
            if name_start < len(source) and source[name_start] == "/":
                name_start += 1
            if name_start < len(source) and source[name_start].isupper():
                tag_end = mdx_tag_end(source, name_start)
                if tag_end is not None:
                    line_end = source.find("\n", tag_end)
                    if line_end == -1:
                        line_end = len(source)
                    if not source[tag_end:line_end].strip():
                        tag_text = source[content_start:tag_end]
                        name_match = re.match(r"</?([A-Z][A-Za-z0-9_.:-]*)", tag_text)
                        if name_match is not None:
                            tags[line_start] = MdxTag(
                                start=line_start,
                                end=tag_end,
                                name=name_match.group(1),
                                indent=content_start - line_start,
                                closing=tag_text.startswith("</"),
                                self_closing=tag_text.rstrip().endswith("/>"),
                            )
                        line_start = tag_end
        next_line = source.find("\n", line_start)
        if next_line == -1:
            break
        line_start = next_line + 1
    return tags


def parser_view(source: str) -> str:
    """Expose Markdown nested in Fern JSX in a parser-only source view."""

    masked = list(source)

    # markdown-it-py does not understand MDX comments. Leaving them in place can
    # turn the following reference definitions into paragraph text.
    for comment in re.finditer(r"\{/\*.*?\*/\}", source, flags=re.DOTALL):
        masked[comment.start() : comment.end()] = mask_characters(
            source, comment.start(), comment.end()
        )

    # CommonMark treats an uppercase JSX container as a raw HTML block and does
    # not parse its Markdown children. Blank standalone component tags and
    # remove their structural child indentation in this parser-only view.
    tags = standalone_mdx_tags(source)
    for tag in tags.values():
        masked[tag.start : tag.end] = mask_characters(source, tag.start, tag.end)

    result: list[str] = []
    component_stack: list[MdxTag] = []
    line_start = 0
    while line_start < len(source):
        line_end = source.find("\n", line_start)
        if line_end == -1:
            line_end = len(source)
        else:
            line_end += 1

        tag = tags.get(line_start)
        if tag is not None and tag.closing:
            while component_stack:
                opened = component_stack.pop()
                if opened.name == tag.name:
                    break

        line = "".join(masked[line_start:line_end])
        if tag is None and component_stack:
            desired_indent = component_stack[-1].indent + 2
            removed = 0
            while (
                removed < len(line)
                and removed < desired_indent
                and line[removed] in " \t"
            ):
                removed += 1
            line = line[removed:]
        result.append(line)

        if tag is not None and not tag.closing and not tag.self_closing:
            component_stack.append(tag)
        line_start = line_end

    return "".join(result)


def token_destinations(
    tokens: Iterable[Token], env: dict[str, Any]
) -> list[tuple[str, str]]:
    destinations: list[tuple[str, str]] = []
    for token in walk_tokens(tokens):
        if token.type == "link_open":
            href = token.attrGet("href")
            if href is not None:
                destinations.append((href, "link"))
        elif token.type == "image":
            src = token.attrGet("src")
            if src is not None:
                destinations.append((src, "image"))
        elif token.type == "definition":
            url = token.meta.get("url")
            if url is not None:
                destinations.append((url, "definition"))

    for reference in env.get("references", {}).values():
        href = reference.get("href")
        if href is not None:
            destinations.append((href, "definition"))
    return destinations


def parser_destinations(md: MarkdownIt, source: str) -> list[tuple[str, str]]:
    env: dict[str, Any] = {}
    return token_destinations(md.parse(parser_view(source), env), env)


def confirm_candidates(
    md: MarkdownIt, source: str, candidates: list[Candidate]
) -> list[tuple[str, str]]:
    instrumented = replace_candidate_spans(source, candidates)
    untagged_destinations: list[tuple[str, str]] = []
    for destination, kind in parser_destinations(md, instrumented):
        match = SENTINEL.fullmatch(destination)
        if match is None:
            untagged_destinations.append((destination, kind))
            continue
        candidate_id = int(match.group("id"))
        if 0 <= candidate_id < len(candidates):
            candidates[candidate_id].parser_kinds.add(kind)
    return untagged_destinations


def is_relative_path(destination: str) -> bool:
    if destination.startswith(("#", "/", "\\")):
        return False
    parsed = urlsplit(destination)
    return not parsed.scheme and not parsed.netloc and bool(parsed.path)


def resolved_path(source_path: Path, destination: str) -> Path | None:
    if not is_relative_path(destination):
        return None
    path = unquote(urlsplit(destination).path)
    return (source_path.parent / path).resolve()


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def append_url_suffix(url: str, destination: str) -> str:
    parsed = urlsplit(destination)
    if parsed.query:
        url += f"?{parsed.query}"
    if parsed.fragment:
        url += f"#{parsed.fragment}"
    return url


def github_url(
    path: Path,
    repo_root: Path,
    repository: str,
    ref: str,
    parser_kinds: set[str],
    destination: str,
) -> tuple[str | None, str | None]:
    relative_path = quote(path.relative_to(repo_root).as_posix(), safe="/@:+")
    quoted_ref = quote(ref, safe="/@:+")

    if "image" in parser_kinds:
        if parser_kinds - {"image", "definition"}:
            return None, "destination is shared by image and non-image syntax"
        url = f"https://raw.githubusercontent.com/{repository}/{quoted_ref}/{relative_path}"
        return append_url_suffix(url, destination), None

    object_kind = "tree" if path.is_dir() else "blob"
    url = f"https://github.com/{repository}/{object_kind}/{quoted_ref}/{relative_path}"
    return append_url_suffix(url, destination), None


def line_and_column(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    return line, offset - last_newline


def candidate_report(source: str, candidate: Candidate) -> dict[str, Any]:
    line, column = line_and_column(source, candidate.start)
    return {
        "line": line,
        "column": column,
        "destination": candidate.destination,
        "syntax": candidate.syntax,
    }


def apply_replacements(source: str, replacements: Iterable[Replacement]) -> str:
    rewritten = source
    for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
        if rewritten[replacement.start : replacement.end] != replacement.destination:
            raise ValueError(
                f"source changed before replacement at {replacement.start}:{replacement.end}"
            )
        rewritten = (
            rewritten[: replacement.start]
            + replacement.replacement
            + rewritten[replacement.end :]
        )
    return rewritten


def process_file(
    md: MarkdownIt,
    source_path: Path,
    destination_path: Path,
    docs_root: Path,
    repo_root: Path,
    repository: str,
    ref: str,
) -> dict[str, Any]:
    source = source_path.read_bytes().decode("utf-8")
    candidates = scan_candidates(source)
    untagged_destinations = confirm_candidates(md, source, candidates)

    confirmed = [candidate for candidate in candidates if candidate.parser_kinds]
    missing_mappings = sorted(
        {
            md.normalizeLink(destination)
            for destination, _kind in untagged_destinations
            if (path := resolved_path(source_path, destination)) is not None
            and not is_within(path, docs_root)
        }
    )

    result: dict[str, Any] = {
        "file": str(source_path.relative_to(docs_root)),
        "scannerCandidates": len(candidates),
        "parserConfirmed": len(confirmed),
        "parserRejected": [
            candidate_report(source, candidate)
            for candidate in candidates
            if not candidate.parser_kinds
        ],
        "replacements": [],
        "skippedReplacements": [],
        "unresolved": [],
        "mappingErrors": [],
    }
    if missing_mappings:
        result["mappingErrors"].append(
            {
                "error": "parser found external relative destinations without source spans",
                "destinations": missing_mappings,
            }
        )
        return result
    replacements: list[Replacement] = []
    for candidate in confirmed:
        path = resolved_path(source_path, candidate.destination)
        if path is None or is_within(path, docs_root):
            continue

        line, column = line_and_column(source, candidate.start)
        issue = {
            "line": line,
            "column": column,
            "destination": candidate.destination,
            "parserKinds": sorted(candidate.parser_kinds),
        }
        if not is_within(path, repo_root):
            result["unresolved"].append(
                {**issue, "reason": "destination resolves outside the repository"}
            )
            continue
        if not path.exists():
            result["unresolved"].append(
                {**issue, "reason": "destination does not exist"}
            )
            continue

        replacement_url, error = github_url(
            path,
            repo_root,
            repository,
            ref,
            candidate.parser_kinds,
            candidate.destination,
        )
        if error is not None or replacement_url is None:
            result["unresolved"].append({**issue, "reason": error})
            continue

        replacements.append(
            Replacement(
                candidate.start,
                candidate.end,
                source[candidate.start : candidate.end],
                replacement_url,
                tuple(sorted(candidate.parser_kinds)),
            )
        )
        result["replacements"].append({**issue, "replacement": replacement_url})

    if result["unresolved"]:
        result["skippedReplacements"] = result["replacements"]
        result["replacements"] = []
        return result

    if replacements:
        rewritten = apply_replacements(source, replacements)
        rewritten_destinations = {
            md.normalizeLink(destination)
            for destination, _kind in parser_destinations(md, rewritten)
        }
        missing_rewrites = sorted(
            {md.normalizeLink(replacement.replacement) for replacement in replacements}
            - rewritten_destinations
        )
        if missing_rewrites:
            result["mappingErrors"].append(
                {
                    "error": "rewritten destinations were not recognized by the parser",
                    "destinations": missing_rewrites,
                }
            )
            result["skippedReplacements"] = result["replacements"]
            result["replacements"] = []
        else:
            destination_path.write_bytes(rewritten.encode("utf-8"))

    return result


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output.resolve()
    if not is_within(input_root, repo_root):
        raise SystemExit(f"input must be within repo root: {input_root}")
    if (
        input_root == output_root
        or input_root in output_root.parents
        or output_root in input_root.parents
    ):
        raise SystemExit("input and output trees must not contain one another")

    if output_root.exists():
        if not args.force:
            raise SystemExit(
                f"output already exists: {output_root} (pass --force to replace it)"
            )
        shutil.rmtree(output_root)
    shutil.copytree(input_root, output_root)

    md = build_parser()
    files = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix in MD_EXTENSIONS
    )
    file_results = [
        process_file(
            md,
            source_path,
            output_root / source_path.relative_to(input_root),
            input_root,
            repo_root,
            args.repository,
            args.ref,
        )
        for source_path in files
    ]

    summary = defaultdict(int)
    summary["files"] = len(file_results)
    for result in file_results:
        summary["scannerCandidates"] += result["scannerCandidates"]
        summary["parserConfirmed"] += result["parserConfirmed"]
        summary["parserRejected"] += len(result["parserRejected"])
        summary["replacements"] += len(result["replacements"])
        summary["skippedReplacements"] += len(result["skippedReplacements"])
        summary["unresolved"] += len(result["unresolved"])
        summary["mappingErrors"] += len(result["mappingErrors"])
        if result["replacements"]:
            summary["changedFiles"] += 1

    report = {
        "parser": "markdown-it-py",
        "strategy": "parser-confirmed source-span replacement",
        "input": str(input_root),
        "output": str(output_root),
        "repository": args.repository,
        "ref": args.ref,
        "summary": dict(summary),
        "files": [
            result
            for result in file_results
            if result["replacements"]
            or result["skippedReplacements"]
            or result["parserRejected"]
            or result["unresolved"]
            or result["mappingErrors"]
        ],
    }
    report_text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    print(report_text, end="")

    has_errors = summary["unresolved"] or summary["mappingErrors"]
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

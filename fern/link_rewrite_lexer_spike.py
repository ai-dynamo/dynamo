#!/usr/bin/env -S uv run --script

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# ///

"""Rewrite docs links with a dependency-free, source-preserving Markdown lexer.

The lexer protects code, comments, frontmatter, and HTML/JSX tag spans before
recognizing inline Markdown destinations and reference definitions. It never
renders Markdown and never modifies the input tree.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import string
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlsplit


MD_EXTENSIONS = {".md", ".mdx"}
WHITESPACE = " \t\r\n"
ESCAPABLE = frozenset(string.punctuation)
RAW_CODE_ELEMENT = re.compile(
    r"<(?P<tag>pre|code|script|style)(?:\s[^>]*)?>.*?</(?P=tag)\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    kind: str = field(compare=False)


class SpanSet:
    def __init__(self, spans: Iterable[Span] = ()) -> None:
        ordered = sorted(spans)
        merged: list[Span] = []
        for span in ordered:
            if span.start >= span.end:
                continue
            if merged and span.start <= merged[-1].end:
                previous = merged[-1]
                merged[-1] = Span(
                    previous.start,
                    max(previous.end, span.end),
                    f"{previous.kind}+{span.kind}",
                )
            else:
                merged.append(span)
        self.spans = merged
        self.starts = [span.start for span in merged]

    def covering_end(self, position: int) -> int | None:
        index = bisect_right(self.starts, position) - 1
        if index >= 0:
            span = self.spans[index]
            if span.start <= position < span.end:
                return span.end
        return None

    def overlaps(self, start: int, end: int) -> bool:
        covering = self.covering_end(start)
        if covering is not None:
            return True
        index = bisect_right(self.starts, start)
        return index < len(self.spans) and self.spans[index].start < end


@dataclass
class Candidate:
    start: int
    end: int
    destination: str
    syntax: str
    kinds: set[str]
    label: str | None = None


@dataclass(frozen=True)
class ParsedDestination:
    start: int
    end: int
    destination: str
    next_position: int


@dataclass(frozen=True)
class ParsedInlineLink:
    candidate: Candidate | None
    end: int
    label_end: int


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    destination: str
    replacement: str


@dataclass(frozen=True)
class LexResult:
    candidates: list[Candidate]
    protected_spans: list[Span]
    protected_counts: dict[str, int]


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
        help="fail when a recognized destination cannot be resolved",
    )
    return parser.parse_args()


def is_escaped(source: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and source[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def find_delimited_spans(
    source: str, opener: str, closer: str, kind: str
) -> list[Span]:
    spans: list[Span] = []
    position = 0
    while True:
        start = source.find(opener, position)
        if start == -1:
            break
        close = source.find(closer, start + len(opener))
        end = len(source) if close == -1 else close + len(closer)
        spans.append(Span(start, end, kind))
        position = end
    return spans


def find_frontmatter_span(source: str) -> list[Span]:
    position = 1 if source.startswith("\ufeff") else 0
    first_end = source.find("\n", position)
    if first_end == -1 or source[position:first_end].rstrip("\r") != "---":
        return []
    line_start = first_end + 1
    while line_start < len(source):
        line_end = source.find("\n", line_start)
        if line_end == -1:
            line_end = len(source)
        if source[line_start:line_end].rstrip("\r") in {"---", "..."}:
            end = line_end + 1 if line_end < len(source) else line_end
            return [Span(position, end, "frontmatter")]
        line_start = line_end + 1
    return [Span(position, len(source), "frontmatter")]


def tag_name_end(source: str, position: int) -> int | None:
    if position < len(source) and source[position] == "/":
        position += 1
    if position >= len(source) or not source[position].isalpha():
        return None
    start = position
    while position < len(source) and (
        source[position].isalnum() or source[position] in "_.:-"
    ):
        position += 1
    if start == position:
        return None
    if position < len(source) and source[position] not in " \t\r\n/>":
        return None
    return position


def tag_end(source: str, position: int) -> int | None:
    quote_character: str | None = None
    brace_depth = 0
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


def find_tag_spans(source: str, protected: SpanSet) -> list[Span]:
    spans: list[Span] = []
    position = 0
    while position < len(source):
        covering_end = protected.covering_end(position)
        if covering_end is not None:
            position = covering_end
            continue
        start = source.find("<", position)
        if start == -1:
            break
        covering_end = protected.covering_end(start)
        if covering_end is not None:
            position = covering_end
            continue
        if start > 0 and source[start - 1] == "<":
            position = start + 1
            continue
        previous = start - 1
        while previous >= 0 and source[previous] in WHITESPACE:
            previous -= 1
        if previous >= 0 and source[previous] == "(":
            position = start + 1
            continue
        line_start = source.rfind("\n", 0, start) + 1
        if re.search(r"\][ \t]*:[ \t]*$", source[line_start:start]):
            position = start + 1
            continue
        name_end = tag_name_end(source, start + 1)
        if name_end is None:
            position = start + 1
            continue
        end = tag_end(source, name_end)
        if end is None:
            position = start + 1
            continue
        spans.append(Span(start, end, "html-or-jsx-tag"))
        position = end
    return spans


def fence_start(line: str) -> tuple[str, int, str] | None:
    position = 0
    while position < len(line) and line[position] in " \t":
        position += 1
    while position < len(line) and line[position] == ">":
        position += 1
        if position < len(line) and line[position] == " ":
            position += 1
        while position < len(line) and line[position] in " \t":
            position += 1
    if position >= len(line) or line[position] not in "`~":
        return None
    character = line[position]
    end = position
    while end < len(line) and line[end] == character:
        end += 1
    length = end - position
    if length < 3:
        return None
    remainder = line[end:].rstrip("\r\n")
    if character == "`" and "`" in remainder:
        return None
    return character, length, remainder


def is_fence_close(line: str, character: str, minimum_length: int) -> bool:
    parsed = fence_start(line)
    if parsed is None:
        return False
    found_character, length, remainder = parsed
    return (
        found_character == character
        and length >= minimum_length
        and not remainder.strip()
    )


def find_fence_spans(source: str, protected: SpanSet) -> list[Span]:
    spans: list[Span] = []
    open_start: int | None = None
    open_character = ""
    open_length = 0
    line_start = 0
    while line_start < len(source):
        line_end = source.find("\n", line_start)
        if line_end == -1:
            line_end = len(source)
        else:
            line_end += 1
        line = source[line_start:line_end]
        if open_start is not None:
            if is_fence_close(line, open_character, open_length):
                spans.append(Span(open_start, line_end, "fenced-code"))
                open_start = None
        elif protected.covering_end(line_start) is None:
            parsed = fence_start(line)
            if parsed is not None:
                open_character, open_length, _remainder = parsed
                open_start = line_start
        line_start = line_end
    if open_start is not None:
        spans.append(Span(open_start, len(source), "fenced-code"))
    return spans


def find_inline_code_spans(source: str, protected: SpanSet) -> list[Span]:
    spans: list[Span] = []
    position = 0
    while position < len(source):
        covering_end = protected.covering_end(position)
        if covering_end is not None:
            position = covering_end
            continue
        if source[position] != "`" or is_escaped(source, position):
            position += 1
            continue
        opener_start = position
        while position < len(source) and source[position] == "`":
            position += 1
        opener_length = position - opener_start
        search = position
        line_end = source.find("\n", position)
        if line_end == -1:
            line_end = len(source)
        close_end: int | None = None
        while search < line_end:
            covering_end = protected.covering_end(search)
            if covering_end is not None:
                search = covering_end
                continue
            tick = source.find("`", search, line_end)
            if tick == -1:
                break
            covering_end = protected.covering_end(tick)
            if covering_end is not None:
                search = covering_end
                continue
            run_end = tick
            while run_end < len(source) and source[run_end] == "`":
                run_end += 1
            if run_end - tick == opener_length:
                close_end = run_end
                break
            search = run_end
        if close_end is None:
            continue
        spans.append(Span(opener_start, close_end, "inline-code"))
        position = close_end
    return spans


def build_protected_spans(source: str) -> tuple[SpanSet, dict[str, int]]:
    spans = find_frontmatter_span(source)
    spans += find_delimited_spans(source, "{/*", "*/}", "mdx-comment")
    spans += find_delimited_spans(source, "<!--", "-->", "html-comment")
    spans += [
        Span(match.start(), match.end(), "raw-code-element")
        for match in RAW_CODE_ELEMENT.finditer(source)
    ]

    base = SpanSet(spans)
    spans += find_fence_spans(source, base)
    with_fences = SpanSet(spans)
    spans += find_tag_spans(source, with_fences)
    with_tags = SpanSet(spans)
    spans += find_inline_code_spans(source, with_tags)

    counts: dict[str, int] = defaultdict(int)
    for span in spans:
        counts[span.kind] += 1
    return SpanSet(spans), dict(sorted(counts.items()))


def decode_destination(raw: str) -> str:
    unescaped: list[str] = []
    position = 0
    while position < len(raw):
        if (
            raw[position] == "\\"
            and position + 1 < len(raw)
            and raw[position + 1] in ESCAPABLE
        ):
            position += 1
        unescaped.append(raw[position])
        position += 1
    return html.unescape("".join(unescaped))


def parse_destination(
    source: str, position: int, maximum: int
) -> ParsedDestination | None:
    if position >= maximum:
        return None
    if source[position] == "<":
        start = position + 1
        position = start
        while position < maximum:
            character = source[position]
            if character in "\r\n<":
                return None
            if character == "\\" and position + 1 < maximum:
                position += 2
                continue
            if character == ">":
                if start == position:
                    return None
                raw = source[start:position]
                return ParsedDestination(
                    start, position, decode_destination(raw), position + 1
                )
            position += 1
        return None

    start = position
    parenthesis_depth = 0
    while position < maximum:
        character = source[position]
        if character in WHITESPACE or ord(character) < 0x20 or ord(character) == 0x7F:
            break
        if character == "\\" and position + 1 < maximum:
            position += 2
            continue
        if character == "(":
            parenthesis_depth += 1
            if parenthesis_depth > 32:
                return None
        elif character == ")":
            if parenthesis_depth == 0:
                break
            parenthesis_depth -= 1
        position += 1
    if start == position or parenthesis_depth:
        return None
    raw = source[start:position]
    return ParsedDestination(start, position, decode_destination(raw), position)


def skip_whitespace(source: str, position: int, maximum: int) -> int:
    while position < maximum and source[position] in WHITESPACE:
        position += 1
    return position


def parse_title(source: str, position: int, maximum: int) -> int | None:
    if position >= maximum or source[position] not in {'"', "'", "("}:
        return None
    opener = source[position]
    closer = ")" if opener == "(" else opener
    position += 1
    while position < maximum:
        character = source[position]
        if character == "\\" and position + 1 < maximum:
            position += 2
            continue
        if character == closer:
            return position + 1
        if opener == "(" and character == "(":
            return None
        position += 1
    return None


def bracket_end(
    source: str, start: int, maximum: int, protected: SpanSet
) -> int | None:
    depth = 1
    position = start + 1
    while position < maximum:
        covering_end = protected.covering_end(position)
        if covering_end is not None:
            position = covering_end
            continue
        character = source[position]
        if character == "\\" and position + 1 < maximum:
            position += 2
            continue
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return position
        position += 1
    return None


def parse_inline_link(
    source: str,
    label_start: int,
    image: bool,
    protected: SpanSet,
) -> ParsedInlineLink | None:
    label_end = bracket_end(source, label_start, len(source), protected)
    if label_end is None or label_end + 1 >= len(source):
        return None
    if source[label_end + 1] != "(":
        return None

    position = skip_whitespace(source, label_end + 2, len(source))
    if position < len(source) and source[position] == ")":
        return ParsedInlineLink(None, position + 1, label_end)
    destination = parse_destination(source, position, len(source))
    if destination is None or protected.overlaps(destination.start, destination.end):
        return None

    position = destination.next_position
    whitespace_end = skip_whitespace(source, position, len(source))
    if whitespace_end != position:
        title_end = parse_title(source, whitespace_end, len(source))
        if title_end is not None:
            position = skip_whitespace(source, title_end, len(source))
        else:
            position = whitespace_end
    if position >= len(source) or source[position] != ")":
        return None

    candidate = Candidate(
        destination.start,
        destination.end,
        destination.destination,
        "inline",
        {"image" if image else "link"},
    )
    return ParsedInlineLink(candidate, position + 1, label_end)


def find_nested_image_candidates(
    source: str,
    start: int,
    end: int,
    protected: SpanSet,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    position = start
    while position < end:
        image_start = source.find("![", position, end)
        if image_start == -1:
            break
        if protected.covering_end(image_start) is not None or is_escaped(
            source, image_start
        ):
            position = image_start + 2
            continue
        parsed = parse_inline_link(source, image_start + 1, True, protected)
        if parsed is None or parsed.end > end:
            position = image_start + 2
            continue
        if parsed.candidate is not None:
            candidates.append(parsed.candidate)
        position = parsed.end
    return candidates


def find_inline_candidates(source: str, protected: SpanSet) -> list[Candidate]:
    candidates: list[Candidate] = []
    position = 0
    while position < len(source):
        covering_end = protected.covering_end(position)
        if covering_end is not None:
            position = covering_end
            continue
        image = source.startswith("![", position) and not is_escaped(source, position)
        if image:
            label_start = position + 1
        elif source[position] == "[" and not is_escaped(source, position):
            label_start = position
        else:
            position += 1
            continue
        parsed = parse_inline_link(source, label_start, image, protected)
        if parsed is None:
            position += 1
            continue
        if parsed.candidate is not None:
            candidates.append(parsed.candidate)
        candidates.extend(
            find_nested_image_candidates(
                source,
                label_start + 1,
                parsed.label_end,
                protected,
            )
        )
        position = parsed.end
    return candidates


def normalize_reference(label: str) -> str:
    return " ".join(label.split()).casefold()


def find_reference_candidates(source: str, protected: SpanSet) -> list[Candidate]:
    candidates: list[Candidate] = []
    line_start = 0
    while line_start < len(source):
        line_end = source.find("\n", line_start)
        if line_end == -1:
            line_end = len(source)
        else:
            line_end += 1
        if protected.covering_end(line_start) is not None:
            line_start = line_end
            continue

        position = line_start
        while position < line_end and source[position] in " \t":
            position += 1
        while position < line_end and source[position] == ">":
            position += 1
            if position < line_end and source[position] == " ":
                position += 1
            while position < line_end and source[position] in " \t":
                position += 1
        if position >= line_end or source[position] != "[":
            line_start = line_end
            continue
        label_start = position
        label_end = bracket_end(source, label_start, line_end, protected)
        if (
            label_end is None
            or label_end + 1 >= len(source)
            or source[label_end + 1] != ":"
        ):
            line_start = line_end
            continue

        position = label_end + 2
        while position < len(source) and source[position] in " \t":
            position += 1
        if position < len(source) and source[position] in "\r\n":
            if source.startswith("\r\n", position):
                position += 2
            else:
                position += 1
            while position < len(source) and source[position] in " \t":
                position += 1
        destination = parse_destination(source, position, len(source))
        if destination is None or protected.overlaps(
            destination.start, destination.end
        ):
            line_start = line_end
            continue

        candidate = Candidate(
            destination.start,
            destination.end,
            destination.destination,
            "reference",
            {"definition"},
            normalize_reference(source[label_start + 1 : label_end]),
        )
        candidates.append(candidate)
        line_start = line_end
    return candidates


def assign_reference_kinds(
    source: str, protected: SpanSet, definitions: list[Candidate]
) -> None:
    by_label: dict[str, list[Candidate]] = defaultdict(list)
    for definition in definitions:
        if definition.label:
            by_label[definition.label].append(definition)

    position = 0
    while position < len(source):
        covering_end = protected.covering_end(position)
        if covering_end is not None:
            position = covering_end
            continue
        image = source.startswith("![", position) and not is_escaped(source, position)
        if image:
            label_start = position + 1
        elif source[position] == "[" and not is_escaped(source, position):
            label_start = position
        else:
            position += 1
            continue
        label_end = bracket_end(source, label_start, len(source), protected)
        if label_end is None:
            position += 1
            continue
        if label_end + 1 < len(source) and source[label_end + 1] in "(:":
            position = label_end + 1
            continue

        first_label = source[label_start + 1 : label_end]
        reference_label = first_label
        end = label_end + 1
        if end < len(source) and source[end] == "[":
            second_end = bracket_end(source, end, len(source), protected)
            if second_end is None:
                position = end + 1
                continue
            second_label = source[end + 1 : second_end]
            reference_label = second_label or first_label
            end = second_end + 1

        normalized = normalize_reference(reference_label)
        for definition in by_label.get(normalized, []):
            definition.kinds.add("image" if image else "link")
        position = end


def lex_markdown(source: str) -> LexResult:
    protected, protected_counts = build_protected_spans(source)
    inline = find_inline_candidates(source, protected)
    references = find_reference_candidates(source, protected)
    assign_reference_kinds(source, protected, references)
    candidates = sorted(inline + references, key=lambda candidate: candidate.start)
    previous_end = -1
    for candidate in candidates:
        if candidate.start < previous_end:
            raise ValueError(
                f"overlapping lexer candidates at offsets {candidate.start}:{candidate.end}"
            )
        previous_end = candidate.end
    return LexResult(candidates, protected.spans, protected_counts)


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
    kinds: set[str],
    destination: str,
) -> tuple[str | None, str | None]:
    relative_path = quote(path.relative_to(repo_root).as_posix(), safe="/@:+")
    quoted_ref = quote(ref, safe="/@:+")
    if "image" in kinds:
        if kinds - {"image", "definition"}:
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
    source_path: Path,
    destination_path: Path,
    docs_root: Path,
    repo_root: Path,
    repository: str,
    ref: str,
) -> dict[str, Any]:
    source = source_path.read_bytes().decode("utf-8")
    result: dict[str, Any] = {
        "file": str(source_path.relative_to(docs_root)),
        "lexerCandidates": 0,
        "protectedSpans": 0,
        "protectedByKind": {},
        "replacements": [],
        "skippedReplacements": [],
        "unresolved": [],
        "lexerErrors": [],
    }
    try:
        lexed = lex_markdown(source)
    except ValueError as error:
        result["lexerErrors"].append(str(error))
        return result

    result["lexerCandidates"] = len(lexed.candidates)
    result["protectedSpans"] = len(lexed.protected_spans)
    result["protectedByKind"] = lexed.protected_counts
    replacements: list[Replacement] = []
    for candidate in lexed.candidates:
        path = resolved_path(source_path, candidate.destination)
        if path is None or is_within(path, docs_root):
            continue
        line, column = line_and_column(source, candidate.start)
        issue = {
            "line": line,
            "column": column,
            "destination": candidate.destination,
            "syntax": candidate.syntax,
            "kinds": sorted(candidate.kinds),
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
            candidate.kinds,
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
            candidate.destination for candidate in lex_markdown(rewritten).candidates
        }
        missing = sorted(
            {replacement.replacement for replacement in replacements}
            - rewritten_destinations
        )
        if missing:
            result["lexerErrors"].append(
                {
                    "error": "rewritten destinations were not recognized by the lexer",
                    "destinations": missing,
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

    files = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix in MD_EXTENSIONS
    )
    file_results = [
        process_file(
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
        summary["lexerCandidates"] += result["lexerCandidates"]
        summary["protectedSpans"] += result["protectedSpans"]
        summary["replacements"] += len(result["replacements"])
        summary["skippedReplacements"] += len(result["skippedReplacements"])
        summary["unresolved"] += len(result["unresolved"])
        summary["lexerErrors"] += len(result["lexerErrors"])
        if result["replacements"]:
            summary["changedFiles"] += 1

    report = {
        "strategy": "dependency-free source lexer",
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
            or result["unresolved"]
            or result["lexerErrors"]
        ],
    }
    report_text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    has_errors = summary["unresolved"] or summary["lexerErrors"]
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

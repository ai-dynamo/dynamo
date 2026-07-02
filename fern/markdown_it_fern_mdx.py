# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""markdown-it-py block plugin for Fern's standalone MDX components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token

if TYPE_CHECKING:
    from mdformat.renderer import RenderContext, RenderTreeNode


TAG_START = re.compile(r"<(?P<closing>/)?(?P<name>[A-Z][A-Za-z0-9_.:-]*)")
COMMENT_START = "{/*"
COMMENT_END = "*/}"


@dataclass(frozen=True)
class MdxTag:
    name: str
    start_line: int
    next_line: int
    start: int
    end: int
    closing: bool
    self_closing: bool


def _tag_end(source: str, position: int, maximum: int) -> int | None:
    quote_character: str | None = None
    brace_depth = 0
    while position < maximum:
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
        elif character == ">" and not brace_depth:
            return position + 1
        position += 1
    return None


def _line_after_offset(state: StateBlock, line: int, end_line: int, offset: int) -> int:
    while line < end_line and state.eMarks[line] < offset:
        line += 1
    return min(line + 1, end_line)


def _parse_tag(state: StateBlock, line: int, end_line: int) -> MdxTag | None:
    if line >= end_line or state.is_code_block(line):
        return None
    start = state.bMarks[line] + state.tShift[line]
    match = TAG_START.match(state.src, start, state.eMarks[line])
    if match is None:
        return None
    end = _tag_end(state.src, match.end(), state.eMarks[end_line - 1])
    if end is None:
        return None
    next_line = _line_after_offset(state, line, end_line, end)
    if state.src[end : state.eMarks[next_line - 1]].strip():
        return None
    source = state.src[start:end]
    return MdxTag(
        name=match.group("name"),
        start_line=line,
        next_line=next_line,
        start=start,
        end=end,
        closing=match.group("closing") is not None,
        self_closing=source.rstrip().endswith("/>"),
    )


def _find_closing_tag(
    state: StateBlock, opening: MdxTag, end_line: int
) -> MdxTag | None:
    depth = 1
    line = opening.next_line
    while line < end_line:
        tag = _parse_tag(state, line, end_line)
        if tag is None:
            line += 1
            continue
        if tag.name == opening.name:
            if tag.closing:
                depth -= 1
                if depth == 0:
                    return tag
            elif not tag.self_closing:
                depth += 1
        line = tag.next_line
    return None


def _content_indent(state: StateBlock, start_line: int, end_line: int) -> int:
    indents = [
        state.sCount[line]
        for line in range(start_line, end_line)
        if not state.isEmpty(line)
    ]
    return min(indents, default=state.blkIndent)


def _source_block(
    state: StateBlock, start_line: int, start: int, next_line: int
) -> str:
    line_start = state.bMarks[start_line]
    if not state.src[line_start:start].isspace():
        line_start = start
    return state.src[line_start : state.bMarks[next_line]]


def _push_raw_token(
    state: StateBlock,
    token_type: str,
    nesting: int,
    start_line: int,
    next_line: int,
    content: str,
) -> Token:
    token = state.push(token_type, "", nesting)
    token.map = [start_line, next_line]
    token.content = content
    return token


def _set_indent(token: Token, state: StateBlock, line: int) -> None:
    token.meta["source_indent"] = state.sCount[line]


def _mdx_comment_rule(
    state: StateBlock, start_line: int, end_line: int, silent: bool
) -> bool:
    start = state.bMarks[start_line] + state.tShift[start_line]
    if not state.src.startswith(COMMENT_START, start):
        return False
    end = state.src.find(COMMENT_END, start + len(COMMENT_START))
    if end == -1:
        return False
    next_line = _line_after_offset(state, start_line, end_line, end + len(COMMENT_END))
    if state.src[end + len(COMMENT_END) : state.eMarks[next_line - 1]].strip():
        return False
    if silent:
        return True
    token = _push_raw_token(
        state,
        "fern_mdx_raw",
        0,
        start_line,
        next_line,
        _source_block(state, start_line, start, next_line),
    )
    _set_indent(token, state, start_line)
    state.line = next_line
    return True


def _mdx_component_rule(
    state: StateBlock, start_line: int, end_line: int, silent: bool
) -> bool:
    opening = _parse_tag(state, start_line, end_line)
    if opening is None:
        return False
    if silent:
        return True

    opening_source = _source_block(
        state, opening.start_line, opening.start, opening.next_line
    )
    if opening.closing or opening.self_closing:
        token = _push_raw_token(
            state,
            "fern_mdx_raw",
            0,
            opening.start_line,
            opening.next_line,
            opening_source,
        )
        _set_indent(token, state, opening.start_line)
        state.line = opening.next_line
        return True

    closing = _find_closing_tag(state, opening, end_line)
    if closing is None:
        return False

    open_token = _push_raw_token(
        state,
        "fern_mdx_container_open",
        1,
        opening.start_line,
        opening.next_line,
        opening_source,
    )
    open_token.meta.update(
        name=opening.name,
        source_indent=state.sCount[opening.start_line],
        content_indent=_content_indent(state, opening.next_line, closing.start_line),
    )

    old_parent = state.parentType
    old_line_max = state.lineMax
    old_block_indent = state.blkIndent
    state.parentType = "fern_mdx_container"
    state.lineMax = closing.start_line
    state.blkIndent = _content_indent(state, opening.next_line, closing.start_line)
    state.md.block.tokenize(state, opening.next_line, closing.start_line)
    state.parentType = old_parent
    state.lineMax = old_line_max
    state.blkIndent = old_block_indent

    close_token = _push_raw_token(
        state,
        "fern_mdx_container_close",
        -1,
        closing.start_line,
        closing.next_line,
        _source_block(state, closing.start_line, closing.start, closing.next_line),
    )
    close_token.meta["name"] = closing.name
    state.line = closing.next_line
    return True


def _render_raw(
    _renderer: Any,
    tokens: Sequence[Token],
    index: int,
    _options: Any,
    _env: Any,
) -> str:
    return tokens[index].content


def _mdx_inline_comment_rule(state: StateInline, silent: bool) -> bool:
    if not state.src.startswith(COMMENT_START, state.pos):
        return False
    end = state.src.find(COMMENT_END, state.pos + len(COMMENT_START))
    if end == -1:
        return False
    end += len(COMMENT_END)
    if not silent:
        token = state.push("fern_mdx_raw_inline", "", 0)
        token.content = state.src[state.pos : end]
    state.pos = end
    return True


def _remove_indent(source: str, width: int) -> str:
    prefix = " " * width
    return "\n".join(
        line.removeprefix(prefix) if line else line for line in source.splitlines()
    )


def _indent(source: str, width: int) -> str:
    prefix = " " * width
    return "\n".join(prefix + line if line else line for line in source.splitlines())


def _render_markdown_raw(node: RenderTreeNode, _context: RenderContext) -> str:
    return _remove_indent(node.content, node.meta.get("source_indent", 0))


def _render_markdown_container(node: RenderTreeNode, context: RenderContext) -> str:
    opening_token = node.nester_tokens.opening
    closing_token = node.nester_tokens.closing
    source_indent = opening_token.meta["source_indent"]
    content_indent = opening_token.meta["content_indent"]
    opening = _remove_indent(opening_token.content, source_indent)
    closing = _remove_indent(closing_token.content, source_indent)
    children = "\n\n".join(
        rendered for child in node.children if (rendered := child.render(context))
    )
    parts = [opening]
    if children:
        parts.append(_indent(children, max(0, content_indent - source_indent)))
    parts.append(closing)
    return "\n".join(parts)


RENDERERS = {
    "fern_mdx_raw": _render_markdown_raw,
    "fern_mdx_raw_inline": _render_markdown_raw,
    "fern_mdx_container": _render_markdown_container,
}


def update_mdit(md: MarkdownIt) -> None:
    """Expose the plugin through mdformat's parser-extension interface."""

    md.use(fern_mdx_plugin)


def fern_mdx_plugin(md: MarkdownIt) -> None:
    """Parse standalone uppercase MDX tags and MDX comments as blocks."""

    alternatives = {"alt": ["paragraph", "reference", "blockquote", "list"]}
    md.block.ruler.before(
        "html_block", "fern_mdx_comment", _mdx_comment_rule, alternatives
    )
    md.block.ruler.before(
        "html_block", "fern_mdx_component", _mdx_component_rule, alternatives
    )
    md.inline.ruler.before("text", "fern_mdx_comment", _mdx_inline_comment_rule)
    md.add_render_rule("fern_mdx_raw", _render_raw)
    md.add_render_rule("fern_mdx_raw_inline", _render_raw)
    md.add_render_rule("fern_mdx_container_open", _render_raw)
    md.add_render_rule("fern_mdx_container_close", _render_raw)

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert GitHub-style admonitions to Fern's admonition format.

GitHub admonitions look like:
    > [!NOTE]
    > Useful information that users should know.

    > [!TIP]
    > Helpful advice for doing things better.

    > [!IMPORTANT]
    > Key information users need to know.

    > [!WARNING]
    > Urgent info that needs immediate user attention.

    > [!CAUTION]
    > Advises about risks or negative outcomes.

Fern admonitions look like:
    <Note>This highlights additional context or supplementary information</Note>
    <Tip>This suggests a helpful tip</Tip>
    <Info>This draws attention to important information</Info>
    <Warning>This raises a warning to watch out for</Warning>
    <Error>This indicates a potential error</Error>

This script can be used when syncing docs from the main docs/ folder
to the fern/pages/ folder for the docs-website branch.

Usage:
    # Convert a single file
    python convert_callouts.py input.md output.md

    # Convert a single file in-place
    python convert_callouts.py input.md

    # Convert all markdown files in a directory
    python convert_callouts.py --dir /path/to/pages

    # Convert from stdin to stdout
    cat input.md | python convert_callouts.py -

    # Run tests
    python convert_callouts.py --test
"""

import argparse
import re
import sys
from pathlib import Path

# Mapping from GitHub alert types to Fern admonition tags
# GitHub types: NOTE, TIP, IMPORTANT, WARNING, CAUTION
# Fern types: Info, Warning, Success, Error, Note, Launch, Tip, Check
GITHUB_TO_FERN_MAPPING = {
    "NOTE": "Note",
    "TIP": "Tip",
    "IMPORTANT": "Info",  # IMPORTANT -> Info (draws attention to important info)
    "WARNING": "Warning",
    "CAUTION": "Error",  # CAUTION -> Error (indicates potential error/risk)
}

# Regex pattern to match GitHub-style admonitions
# Matches:
# > [!TYPE]
# > Content line 1
# > Content line 2
# (ends at a blank line or non-blockquote line)
GITHUB_ADMONITION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)>[ \t]*\[!(?P<type>NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n"
    r"(?P<content>(?:(?P=indent)>[ \t]*.*\n?)+)",
    re.MULTILINE | re.IGNORECASE,
)


def extract_blockquote_content(content: str, indent: str) -> str:
    """
    Extract the actual content from blockquote lines.

    Args:
        content: The raw blockquote content (lines starting with '>')
        indent: The leading indentation before '>'

    Returns:
        The content without blockquote markers, preserving internal formatting.
    """
    lines = content.split("\n")
    extracted_lines = []

    for line in lines:
        # Remove the indent and blockquote marker
        if line.startswith(indent + ">"):
            # Remove indent + '>' and optional single space after
            stripped = line[len(indent) + 1 :]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            extracted_lines.append(stripped)
        elif line.strip() == "":
            # Empty line ends the blockquote
            break
        else:
            # Non-blockquote line ends the content
            break

    # Join and strip trailing whitespace, but preserve internal newlines
    result = "\n".join(extracted_lines).rstrip()
    return result


def convert_single_admonition(match: re.Match) -> str:
    """
    Convert a single GitHub admonition match to Fern format.

    Args:
        match: A regex match object containing the admonition

    Returns:
        The converted Fern-style admonition
    """
    indent = match.group("indent")
    alert_type = match.group("type").upper()
    raw_content = match.group("content")

    # Get the Fern tag for this alert type
    fern_tag = GITHUB_TO_FERN_MAPPING.get(alert_type, "Note")

    # Extract the actual content from blockquote lines
    content = extract_blockquote_content(raw_content, indent)

    # Handle multi-line content
    # For Fern, we can either:
    # 1. Keep it on one line (for short content)
    # 2. Use multi-line format (for longer content)
    content_lines = content.split("\n")

    if len(content_lines) == 1 and len(content) < 100:
        # Single short line - keep on one line
        return f"{indent}<{fern_tag}>{content}</{fern_tag}>\n"
    else:
        # Multi-line content - format with proper indentation
        # Fern supports multi-line content within the tags
        formatted_content = "\n".join(content_lines)
        return f"{indent}<{fern_tag}>\n{formatted_content}\n{indent}</{fern_tag}>\n"


def convert_admonitions(text: str) -> str:
    """
    Convert all GitHub-style admonitions in the text to Fern format.

    Args:
        text: The markdown text containing GitHub admonitions

    Returns:
        The text with admonitions converted to Fern format
    """
    return GITHUB_ADMONITION_PATTERN.sub(convert_single_admonition, text)


def process_file(input_path: Path, output_path: Path | None = None) -> None:
    """
    Process a single markdown file, converting admonitions.

    Args:
        input_path: Path to the input file
        output_path: Path to the output file (None for in-place)
    """
    content = input_path.read_text(encoding="utf-8")
    converted = convert_admonitions(content)

    if output_path is None:
        output_path = input_path

    output_path.write_text(converted, encoding="utf-8")


def process_directory(dir_path: Path, recursive: bool = True) -> int:
    """
    Process all markdown files in a directory.

    Args:
        dir_path: Path to the directory
        recursive: Whether to process subdirectories

    Returns:
        Number of files processed
    """
    pattern = "**/*.md" if recursive else "*.md"
    files = list(dir_path.glob(pattern))
    count = 0

    for file_path in files:
        print(f"Processing: {file_path}")
        process_file(file_path)
        count += 1

    return count


def run_tests():
    """Run all test cases for the convert_admonitions function."""
    import textwrap

    passed = 0
    failed = 0

    def test(name: str, input_text: str, expected: str):
        nonlocal passed, failed
        result = convert_admonitions(input_text)
        if result == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    Input:\n{textwrap.indent(repr(input_text), '      ')}")
            print(f"    Expected:\n{textwrap.indent(repr(expected), '      ')}")
            print(f"    Got:\n{textwrap.indent(repr(result), '      ')}")
            failed += 1

    print("Running tests...\n")

    # Test 1: Simple NOTE conversion (short content -> single line)
    test(
        "Simple NOTE - single line",
        "> [!NOTE]\n> This is a note.\n",
        "<Note>This is a note.</Note>\n",
    )

    # Test 2: Simple TIP conversion
    test(
        "Simple TIP - single line",
        "> [!TIP]\n> This is a tip.\n",
        "<Tip>This is a tip.</Tip>\n",
    )

    # Test 3: IMPORTANT -> Info mapping
    test(
        "IMPORTANT -> Info mapping",
        "> [!IMPORTANT]\n> This is important.\n",
        "<Info>This is important.</Info>\n",
    )

    # Test 4: WARNING conversion
    test(
        "WARNING conversion",
        "> [!WARNING]\n> This is a warning.\n",
        "<Warning>This is a warning.</Warning>\n",
    )

    # Test 5: CAUTION -> Error mapping
    test(
        "CAUTION -> Error mapping",
        "> [!CAUTION]\n> This is a caution.\n",
        "<Error>This is a caution.</Error>\n",
    )

    # Test 6: Multi-line content (should use multi-line format)
    test(
        "Multi-line content",
        "> [!NOTE]\n> Line one.\n> Line two.\n",
        "<Note>\nLine one.\nLine two.\n</Note>\n",
    )

    # Test 7: Long single line (>100 chars -> multi-line format)
    long_content = "A" * 101
    test(
        "Long single line -> multi-line format",
        f"> [!NOTE]\n> {long_content}\n",
        f"<Note>\n{long_content}\n</Note>\n",
    )

    # Test 8: Case insensitivity
    test(
        "Case insensitivity (lowercase)",
        "> [!note]\n> Lowercase note.\n",
        "<Note>Lowercase note.</Note>\n",
    )

    test(
        "Case insensitivity (mixed case)",
        "> [!NoTe]\n> Mixed case note.\n",
        "<Note>Mixed case note.</Note>\n",
    )

    # Test 9: Indented admonition
    test(
        "Indented admonition",
        "  > [!NOTE]\n  > Indented note.\n",
        "  <Note>Indented note.</Note>\n",
    )

    # Test 10: Multiple admonitions in one text
    test(
        "Multiple admonitions",
        "> [!NOTE]\n> First note.\n\nSome text.\n\n> [!TIP]\n> A tip.\n",
        "<Note>First note.</Note>\n\nSome text.\n\n<Tip>A tip.</Tip>\n",
    )

    # Test 11: Admonition with markdown formatting
    test(
        "Admonition with markdown formatting",
        "> [!NOTE]\n> This has **bold** and `code`.\n",
        "<Note>This has **bold** and `code`.</Note>\n",
    )

    # Test 12: Admonition with link
    test(
        "Admonition with link",
        "> [!TIP]\n> See [the docs](https://example.com).\n",
        "<Tip>See [the docs](https://example.com).</Tip>\n",
    )

    # Test 13: Empty content after type
    test(
        "Content on same line as blockquote marker",
        "> [!NOTE]\n>\n",
        "<Note></Note>\n",
    )

    # Test 14: Content with extra spaces
    test(
        "Content with leading space preserved",
        "> [!NOTE]\n>  Two spaces before.\n",
        "<Note> Two spaces before.</Note>\n",
    )

    # Test 15: No conversion needed (not an admonition)
    test(
        "Regular blockquote (no conversion)",
        "> This is just a regular blockquote.\n",
        "> This is just a regular blockquote.\n",
    )

    # Test 16: Admonition in middle of document
    test(
        "Admonition in middle of document",
        "# Header\n\nSome paragraph.\n\n> [!WARNING]\n> Be careful!\n\nMore text.\n",
        "# Header\n\nSome paragraph.\n\n<Warning>Be careful!</Warning>\n\nMore text.\n",
    )

    # Test 17: Tab-indented admonition
    test(
        "Tab-indented admonition",
        "\t> [!NOTE]\n\t> Tab indented.\n",
        "\t<Note>Tab indented.</Note>\n",
    )

    # Test 18: Multi-line with blank line in content (ends at blank)
    test(
        "Multi-line ending at blank line",
        "> [!NOTE]\n> Line one.\n> Line two.\n\nAfter.\n",
        "<Note>\nLine one.\nLine two.\n</Note>\n\nAfter.\n",
    )

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert GitHub-style admonitions to Fern format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path, or '-' for stdin. If --dir is used, this is ignored.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file path. If omitted, modifies input in-place (except for stdin).",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        help="Process all markdown files in the specified directory",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't process subdirectories when using --dir",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run test cases",
    )

    args = parser.parse_args()

    if args.test:
        # Run tests
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.dir:
        # Process directory
        if not args.dir.is_dir():
            print(f"Error: {args.dir} is not a directory", file=sys.stderr)
            sys.exit(1)

        count = process_directory(args.dir, recursive=not args.no_recursive)
        print(f"Processed {count} file(s)")

    elif args.input == "-" or args.input is None:
        # Read from stdin, write to stdout
        content = sys.stdin.read()
        converted = convert_admonitions(content)
        sys.stdout.write(converted)

    else:
        # Process single file
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"Error: {input_path} is not a file", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        process_file(input_path, output_path)

        if output_path:
            print(f"Converted: {input_path} -> {output_path}")
        else:
            print(f"Converted: {input_path} (in-place)")


if __name__ == "__main__":
    main()

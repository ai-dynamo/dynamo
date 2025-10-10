#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dockerfile Generator from Jinja2 Templates

This script processes all .j2 template files in the current directory and generates
corresponding Dockerfiles in the specified output directory (/tmp by default).

Usage:
    python gen_dockerfiles.py [--output-dir OUTPUT_DIR] [--compare-ignore-whitespaces] [--compare-strict]

Options:
    --output-dir DIR                Output directory for generated files (default: /tmp)
    --compare-ignore-whitespaces    Compare generated files with originals, ignoring whitespace differences
    --compare-strict                Compare generated files with originals, including all whitespace differences

The script will:
1. Find all *.j2 template files in the current directory
2. Render them using Jinja2 (currently just copies content, ready for template logic)
3. Generate corresponding files without .j2 extension in output directory
4. Optionally compare with original files in parent directory
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def normalize_content_for_comparison(
    content: str, strip_whitespace: bool = True
) -> str:
    """Normalize content by removing TEMPLATE comments and optionally whitespace for comparison.

    Args:
        content: The content to normalize
        strip_whitespace: If True, strip all whitespace and blank lines (default behavior)
                         If False, preserve whitespace but still remove TEMPLATE comments

    Returns:
        Normalized content string
    """
    lines = []
    for line in content.split("\n"):
        # Always skip TEMPLATE comment lines
        if line.strip().startswith("# TEMPLATE:"):
            continue

        if strip_whitespace:
            # Strip all whitespace from each line (including tabs and spaces)
            stripped = line.strip()
            # Skip empty lines when stripping whitespace
            if stripped:
                lines.append(stripped)
        else:
            # Preserve the line as-is (including whitespace)
            lines.append(line)

    return "\n".join(lines)


def compare_files(
    original_path: Path,
    generated_path: Path,
    strict_whitespace: bool = False,
    show_differences: bool = False,
) -> tuple[bool, str]:
    """Compare two files with configurable whitespace handling.

    Args:
        original_path: Path to original file
        generated_path: Path to generated file
        strict_whitespace: If True, include all whitespace differences in comparison
                          If False, ignore whitespace differences
        show_differences: If True, return the diff output

    Returns:
        tuple: (files_match, diff_output or summary)
    """
    try:
        # Read both files
        if not original_path.exists():
            return False, f"Original file not found: {original_path}"

        if not generated_path.exists():
            return False, f"Generated file not found: {generated_path}"

        with open(original_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        with open(generated_path, "r", encoding="utf-8") as f:
            generated_content = f.read()

        # Normalize content based on comparison mode
        strip_whitespace = not strict_whitespace
        original_normalized = normalize_content_for_comparison(
            original_content, strip_whitespace=strip_whitespace
        )
        generated_normalized = normalize_content_for_comparison(
            generated_content, strip_whitespace=strip_whitespace
        )

        # Determine success message based on comparison mode
        if strict_whitespace:
            success_msg = "Files are identical (ignoring TEMPLATE comments only)"
            diff_msg_prefix = (
                "Content differs (strict comparison, ignoring TEMPLATE comments only)"
            )
            file_suffix = ".filtered"
        else:
            success_msg = "Files are identical (ignoring whitespace and blank lines)"
            diff_msg_prefix = (
                "Content differs (after normalizing whitespace and blank lines)"
            )
            file_suffix = ".normalized"

        if original_normalized == generated_normalized:
            return True, success_msg

        # If not identical, show diff for debugging
        if show_differences:
            # Create temporary files for better diff output
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=file_suffix, delete=False
            ) as orig_temp:
                orig_temp.write(original_normalized)
                orig_temp_path = orig_temp.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=file_suffix, delete=False
            ) as gen_temp:
                gen_temp.write(generated_normalized)
                gen_temp_path = gen_temp.name

            try:
                cmd = [
                    "diff",
                    "-u",
                    "--label",
                    f"original/{original_path.name}",
                    "--label",
                    f"generated/{generated_path.name}",
                    orig_temp_path,
                    gen_temp_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 1:
                    return (
                        False,
                        f"{diff_msg_prefix}\n{result.stdout}",
                    )
                else:
                    return (
                        False,
                        diff_msg_prefix,
                    )
            finally:
                # Clean up temporary files
                os.unlink(orig_temp_path)
                os.unlink(gen_temp_path)
        else:
            return (
                False,
                diff_msg_prefix,
            )

    except Exception as e:
        return False, f"Error comparing files: {e}"


def compare_files_strict(
    original_path: Path, generated_path: Path, show_differences: bool = False
) -> tuple[bool, str]:
    """Compare two files strictly including all whitespace differences."""
    return compare_files(
        original_path,
        generated_path,
        strict_whitespace=True,
        show_differences=show_differences,
    )


def compare_files_ignore_whitespace(
    original_path: Path, generated_path: Path, show_differences: bool = False
) -> tuple[bool, str]:
    """Compare two files ignoring whitespace differences."""
    return compare_files(
        original_path,
        generated_path,
        strict_whitespace=False,
        show_differences=show_differences,
    )


def find_template_files(templates_dir: Path) -> list[Path]:
    """Find all .j2 template files in the templates directory."""
    all_templates = list(templates_dir.glob("*.j2"))

    # Filter out template fragments that don't have corresponding original files
    # These are identified by having underscores in their names (e.g., Dockerfile._dev_utils.j2)
    filtered_templates = []
    for template in all_templates:
        # Skip files that have underscore after the last dot (template fragments/includes)
        # e.g., Dockerfile._dev_utils.j2 but not Dockerfile.trtllm_prebuilt.j2
        if "._" in template.name:
            print(f"Skipping template fragment: {template.name}")
        else:
            filtered_templates.append(template)

    return filtered_templates


def generate_dockerfile_from_template(
    template_path: Path, output_path: Path, env: Environment
) -> None:
    """Generate a Dockerfile from a Jinja2 template."""
    print(f"Generating {output_path.name} from {template_path.name}...")

    try:
        # Load and render the template
        template = env.get_template(template_path.name)
        content = template.render()

        # Generate the original template path relative to container/
        original_file_path = f"container/{output_path.name}"

        # Add template header comments
        template_header = f"""# TEMPLATE: This is auto-generated, please modify the original file at {original_file_path} instead of
# TEMPLATE: this file, then run container/templates/gen_dockerfiles.py to generate this file.
"""

        # Combine header with content
        final_content = template_header + content

        # Write the generated content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"Generated: {output_path}")

    except Exception as e:
        print(f"Error generating {output_path.name}: {e}")
        sys.exit(1)


def main():
    """Generate Dockerfiles from Jinja2 templates."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Dockerfiles from Jinja2 templates"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp",
        help="Output directory for generated files (default: /tmp)",
    )
    parser.add_argument(
        "--compare-ignore-whitespaces",
        action="store_true",
        help="Compare generated files in output dir with originals in container/, ignoring whitespace differences",
    )
    parser.add_argument(
        "--compare-strict",
        action="store_true",
        help="Compare generated files with originals in container/, including all whitespace differences",
    )
    parser.add_argument(
        "--show-differences",
        action="store_true",
        help="Show detailed differences when comparing files",
    )
    args = parser.parse_args()

    # Get the script directory (container/templates/)
    script_dir = Path(__file__).parent
    templates_dir = script_dir
    output_dir = Path(args.output_dir)

    # Original files are in the parent directory (container/)
    original_dir = script_dir.parent

    # Verify templates directory exists
    if not templates_dir.exists():
        print(f"Error: Templates directory not found: {templates_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=False,
        lstrip_blocks=True,
    )

    # Find all template files
    template_files = find_template_files(templates_dir)

    if not template_files:
        print(f"No .j2 template files found in {templates_dir}")
        sys.exit(1)

    print(f"Found {len(template_files)} template files")

    try:
        # Generate Dockerfiles from templates
        generated_files = []
        for template_path in template_files:
            # Generate output filename by removing .j2 extension
            output_filename = template_path.stem  # This removes the .j2 extension
            output_path = output_dir / output_filename

            generate_dockerfile_from_template(template_path, output_path, env)
            generated_files.append((template_path, output_path))

        print(
            f"\nSuccess! Generated {len(generated_files)} Dockerfiles in {output_dir}"
        )
        for _, output_path in generated_files:
            print(f"- {output_path.name}")

        # Compare with originals if requested
        if args.compare_ignore_whitespaces or args.compare_strict:
            print("\n" + "=" * 80)
            if args.compare_strict:
                print(
                    f"COMPARISON: {output_dir}/* vs {original_dir}/* (STRICT - INCLUDING WHITESPACE)"
                )
            else:
                print(
                    f"COMPARISON: {output_dir}/* vs {original_dir}/* (WHITESPACE DIFFERENCES IGNORED)"
                )
            print("=" * 80)

            all_identical = True
            for template_path, generated_path in generated_files:
                # Find corresponding original file
                original_filename = generated_path.name
                original_path = original_dir / original_filename

                if original_path.exists():
                    if args.compare_strict:
                        files_match, diff_output = compare_files_strict(
                            original_path, generated_path, args.show_differences
                        )
                    else:
                        files_match, diff_output = compare_files_ignore_whitespace(
                            original_path, generated_path, args.show_differences
                        )

                    if files_match:
                        status = "✅ IDENTICAL"
                        print(f"{original_filename}: {status}")
                    else:
                        all_identical = False
                        if args.show_differences and "\n" in diff_output:
                            lines = diff_output.split("\n", 1)
                            status = f"❌ DIFFERENT ({lines[0]})"
                            print(f"{original_filename}: {status}")
                            if len(lines) > 1:
                                print(lines[1])
                        else:
                            status = f"❌ DIFFERENT ({diff_output})"
                            print(f"{original_filename}: {status}")
                else:
                    print(
                        f"{original_filename}: ⚠️  Original not found at {original_path}"
                    )

            if all_identical:
                print(
                    f"\n🎉 All generated files in {output_dir}/ are identical to their originals in {original_dir}/!"
                )
            else:
                comparison_type = (
                    "strict comparison"
                    if args.compare_strict
                    else "ignoring whitespace"
                )
                print(
                    f"\n⚠️  Some files differ between {output_dir}/ and {original_dir}/ ({comparison_type})"
                )

            if args.compare_strict:
                print(
                    f"\nNote: Strict comparison between {output_dir}/* and {original_dir}/* includes all whitespace differences (ignoring TEMPLATE comments only)"
                )
            else:
                print(
                    f"\nNote: Comparison between {output_dir}/* and {original_dir}/* ignores all whitespace differences, blank lines, and TEMPLATE comments"
                )

    except Exception as e:
        print(f"Error generating Dockerfiles: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate devcontainer.json files for different frameworks using Jinja2 template.
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def main():
    """Generate devcontainer.json files for different frameworks."""
    # Define the frameworks to generate
    frameworks = ["vllm", "sglang", "trtllm"]

    # Get the current directory (where this script is located)
    script_dir = Path(__file__).parent

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(script_dir))
    template = env.get_template("devcontainer.json.j2")

    # Generate devcontainer.json for each framework
    for framework in frameworks:
        # Create the target directory
        target_dir = script_dir / framework
        target_dir.mkdir(exist_ok=True)

        # Render the template with framework-specific values
        current_year = datetime.now().year
        rendered_content = template.render(
            framework=framework, current_year=current_year
        )

        # Add auto-generated warning to the beginning of the file
        warning_comment = """// AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY
// This file was generated from devcontainer.json.j2
// To make changes, edit the .j2 template and run gen_devcontainer_json.py
"""

        # Insert warning after the opening brace
        lines = rendered_content.split("\n")
        if lines[0].strip() == "{":
            lines.insert(1, warning_comment.rstrip())
            rendered_content = "\n".join(lines)

        # Write the rendered content to the target file
        target_file = target_dir / "devcontainer.json"
        with open(target_file, "w") as f:
            f.write(rendered_content)

        print(f"Generated {target_file}")

    print(
        f"Successfully generated devcontainer.json files for: {', '.join(frameworks)}"
    )


if __name__ == "__main__":
    main()

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
Script to convert Go struct definitions from v1beta1 DGDR types to Python Pydantic models.

This script parses the Go file containing Kubernetes CRD type definitions and generates
corresponding Pydantic models that can be used in Python code (e.g., in the profiler).

Usage:
    python generate_pydantic_from_go.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Types that should be IMPORTED rather than re-emitted.
# Maps Go type name → (Python import path, Python name).
# Planner-specific types are the canonical hand-written source of truth.
_IMPORT_OVERRIDES: dict[str, tuple[str, str]] = {
    "PlannerPreDeploymentSweepMode": (
        "dynamo.planner.utils.planner_config",
        "PlannerPreDeploymentSweepMode",
    ),
    "PlannerConfig": (
        "dynamo.planner.utils.planner_config",
        "PlannerConfig",
    ),
}

# Per-struct docstring overrides for cases where the Python docstring should differ
# from the Go comment (e.g. Python-specific mutual-exclusivity documentation).
_STRUCT_DOCSTRINGS: dict = {
    "SLASpec": (
        "Service-level agreement targets.\n\n"
        "    Provide exactly one of:\n\n"
        "    - ``ttft`` + ``itl``: explicit latency targets (default: 2000 ms / 30 ms)\n"
        "    - ``e2eLatency``: end-to-end latency target\n"
        "    - ``optimizationType``: high-level objective without explicit numeric targets"
    ),
}

# Extra Python code (validators, etc.) appended after the generated field list for
# specific structs. Cannot be expressed as Go kubebuilder markers.
_STRUCT_EXTRAS: dict = {
    "SLASpec": """\
    @model_validator(mode="after")
    def _validate_sla_options(self) -> "SLASpec":
        \"\"\"Ensure at most one SLA mode is active.\"\"\"
        has_ttft_itl = self.ttft is not None and self.itl is not None
        has_e2e = self.e2eLatency is not None
        has_opt = self.optimizationType is not None
        options_count = sum([has_ttft_itl, has_e2e, has_opt])
        if options_count > 1:
            raise ValueError(
                "SLA must specify exactly one of: (ttft and itl), e2eLatency, "
                "or optimizationType \u2014 not multiple."
            )
        if (self.ttft is not None) != (self.itl is not None):
            raise ValueError("ttft and itl must both be provided together.")
        return self
""",
}

# Per-field Python type overrides.  Maps (StructName, json_field_name) → Python type string.
# Used when the Go type (e.g. *runtime.RawExtension) should map to a richer Python type
# rather than the generic Dict[str, Any].
_FIELD_TYPE_OVERRIDES: dict[tuple[str, str], str] = {
    # FeaturesSpec.planner is opaque in Go but strongly typed in Python.
    ("FeaturesSpec", "planner"): "Optional[PlannerConfig]",
}

_SPDX_HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License.\
"""


def _resolve_repo_root(start: Path) -> Path:
    """Return the repository root via git, falling back to go.mod traversal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=start,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Fallback: walk up until we find go.mod
    p = start
    while p != p.parent:
        if (p / "go.mod").exists():
            return p
        p = p.parent
    return start


@dataclass
class GoField:
    """Represents a field in a Go struct"""

    name: str
    go_type: str
    json_tag: str
    comment: str
    is_optional: bool
    is_pointer: bool
    default: Optional[str] = field(default=None)


@dataclass
class GoStruct:
    """Represents a Go struct definition"""

    name: str
    fields: List[GoField]
    comment: str


@dataclass
class GoEnum:
    """Represents a Go enum (const block)"""

    name: str
    base_type: str
    values: List[Tuple[str, str]]  # (const_name, const_value)
    comment: str


class GoToPydanticConverter:
    """Converts Go structs to Pydantic models"""

    # Type mapping from Go to Python
    # RUF012: use a plain dict (not ClassVar) — instantiated once per instance
    TYPE_MAP: dict = {
        "string": "str",
        "int": "int",
        "int32": "int",
        "int64": "int",
        "float64": "float",
        "bool": "bool",
        "metav1.ObjectMeta": "Dict[str, Any]",  # Simplified
        "metav1.TypeMeta": "Dict[str, Any]",  # Simplified
        "metav1.Condition": "Dict[str, Any]",  # Simplified
        "runtime.RawExtension": "Dict[str, Any]",
        "batchv1.JobSpec": "Dict[str, Any]",
        "corev1.ResourceRequirements": "Dict[str, Any]",
        "corev1.Toleration": "Dict[str, Any]",
        "apiextensionsv1.JSON": "Any",
    }

    def __init__(self):
        self.structs: List[GoStruct] = []
        self.enums: List[GoEnum] = []

    def parse_go_file(self, file_path: Path) -> None:
        """Parse Go file and extract struct and enum definitions"""
        content = file_path.read_text()

        # Extract enum definitions (const blocks with string type)
        self._parse_enums(content)

        # Extract struct definitions
        self._parse_structs(content)

    def _parse_enums(self, content: str) -> None:
        """Parse Go const blocks that define enum types"""
        # Pattern: // Comment\n// +kubebuilder:validation:Enum=val1;val2\ntype Name string
        enum_pattern = r"(?://.*\n)*// \+kubebuilder:validation:Enum=([^\n]+)\ntype\s+(\w+)\s+string"

        for match in re.finditer(enum_pattern, content):
            enum_values_str = match.group(1)
            enum_name = match.group(2)

            # Extract comment — stop at blank // lines (avoid lifecycle steps etc.)
            lines_before = content[: match.start()].split("\n")
            comment_lines = []
            for line in reversed(lines_before[-10:]):  # Look at last 10 lines
                stripped = line.strip()
                if stripped == "//":
                    break  # blank comment line — stop collecting
                if stripped.startswith("//") and "kubebuilder" not in stripped:
                    comment_lines.insert(0, stripped.lstrip("/ ").strip())
                elif stripped and not stripped.startswith("//"):
                    break

            # Parse enum values from kubebuilder annotation (fallback)
            enum_values_fallback = [v.strip() for v in enum_values_str.split(";")]

            # Try to find individual const definitions for this enum type.
            # We search line-by-line rather than requiring a contiguous block
            # (blocks may contain comments between entries or mix multiple types).
            individual_pattern = (
                rf"(?m)^\s*{enum_name}(\w+)\s+{enum_name}\s+=\s+\"([^\"]+)\""
            )
            values: List[Tuple[str, str]] = []
            for m in re.finditer(individual_pattern, content):
                const_name = m.group(1)
                const_value = m.group(2)
                # Sanitise Python reserved words
                if const_name in (
                    "None",
                    "True",
                    "False",
                    "pass",
                    "return",
                    "class",
                    "def",
                ):
                    const_name = f"{const_name}_"
                values.append((const_name, const_value))

            if not values:
                # Fallback: derive names from kubebuilder annotation values
                values = [
                    (v.upper().replace("-", "_"), v) for v in enum_values_fallback
                ]

            self.enums.append(
                GoEnum(
                    name=enum_name,
                    base_type="string",
                    values=values,
                    comment=" ".join(comment_lines),
                )
            )

    def _parse_structs(self, content: str) -> None:
        """Parse Go struct definitions"""
        struct_pattern = r"type\s+(\w+)\s+struct\s+\{"

        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            start_pos = match.end()

            # Find matching closing brace by counting braces
            brace_count = 1
            pos = start_pos
            while brace_count > 0 and pos < len(content):
                if content[pos] == "{":
                    brace_count += 1
                elif content[pos] == "}":
                    brace_count -= 1
                pos += 1

            struct_body = content[start_pos : pos - 1]

            # Skip List types
            if struct_name.endswith("List"):
                continue

            # Extract the struct docstring using a forward scan.
            #
            # A backward scan fails when kubebuilder markers appear between the
            # description and the type declaration (e.g. DynamoGraphDeploymentRequest),
            # because the blank // separator between the description and later content
            # (lifecycle steps, markers) causes an early stop before any text is collected.
            #
            # Algorithm:
            #   1. Walk backward to find the start of the contiguous comment block.
            #   2. Walk forward from that start, skipping kubebuilder lines, and collect
            #      lines until the first blank // (end of the first paragraph).
            lines_before = content[: match.start()].split("\n")

            # Step 1: find the index just after the last non-comment line before the block
            comment_start_idx = 0
            for i, line in enumerate(reversed(lines_before)):
                stripped = line.strip()
                if stripped and not stripped.startswith("//"):
                    comment_start_idx = len(lines_before) - i
                    break

            # Step 2: forward scan — collect the first paragraph, skip markers
            comment_lines: List[str] = []
            for line in lines_before[comment_start_idx:]:
                stripped = line.strip()
                if stripped == "//":
                    if comment_lines:  # blank line ends the first paragraph
                        break
                elif (
                    stripped.startswith("//")
                    and "kubebuilder" not in stripped
                    and "EDIT THIS FILE" not in stripped
                ):
                    comment_lines.append(stripped.lstrip("/ ").strip())
                # kubebuilder markers and empty lines: skip, keep scanning

            # Parse fields
            fields = self._parse_struct_fields(struct_body)

            self.structs.append(
                GoStruct(
                    name=struct_name, fields=fields, comment=" ".join(comment_lines)
                )
            )

    def _parse_struct_fields(self, struct_body: str) -> List[GoField]:
        """Parse fields from struct body"""
        fields = []

        lines = struct_body.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Collect consecutive comment lines (including single-line comments)
            comment_lines: List[str] = []
            default_value: Optional[str] = None
            while line.startswith("//"):
                # Capture kubebuilder:default (affects both Go API and Python default)
                kb_match = re.search(r"\+kubebuilder:default=(\S+)", line)
                # Capture +python-default (Python-only; does not affect the Go/k8s API)
                py_match = re.search(r"\+python-default=(\S+)", line)
                if kb_match:
                    default_value = kb_match.group(1)
                elif py_match:
                    default_value = py_match.group(1)
                elif (
                    "kubebuilder" not in line
                    and "+optional" not in line.lower()
                    and "+python-default" not in line
                ):
                    comment_lines.append(line.lstrip("/ ").strip())
                i += 1
                if i >= len(lines):
                    break
                line = lines[i].strip()

            # Now line should be a field definition or empty
            if not line:
                i += 1
                continue

            # Pattern: FieldName type `json:"jsonName,omitempty"`
            field_pattern = r'(\w+)\s+([\w\.\*\[\]]+)\s+`json:"([^"]+)"`'
            match = re.match(field_pattern, line)

            if match:
                field_type = match.group(2)
                json_tag_full = match.group(3)

                json_parts = json_tag_full.split(",")
                json_name = json_parts[0]
                is_optional = "omitempty" in json_parts or ",inline" in json_tag_full

                is_pointer = field_type.startswith("*")
                if is_pointer:
                    field_type = field_type[1:]

                # Skip inline fields (metav1.TypeMeta, metav1.ObjectMeta)
                if ",inline" in json_tag_full:
                    i += 1
                    continue

                fields.append(
                    GoField(
                        name=json_name,
                        go_type=field_type,
                        json_tag=json_name,
                        comment=" ".join(comment_lines),
                        is_optional=is_optional,
                        is_pointer=is_pointer,
                        default=default_value,
                    )
                )

            i += 1

        return fields

    def _go_type_to_python(
        self, go_type: str, is_pointer: bool, is_optional: bool
    ) -> str:
        """Convert Go type to Python type hint"""
        # Handle array types
        if go_type.startswith("[]"):
            inner_type = go_type[2:]
            python_inner = self._go_type_to_python(inner_type, False, False)
            result = f"List[{python_inner}]"
            if is_optional:
                return f"Optional[{result}]"
            return result

        # Handle map types
        if go_type.startswith("map["):
            map_match = re.match(r"map\[(\w+)\](.+)", go_type)
            if map_match:
                key_type = self.TYPE_MAP.get(map_match.group(1), "str")
                val_type = self._go_type_to_python(map_match.group(2), False, False)
                result = f"Dict[{key_type}, {val_type}]"
                if is_optional:
                    return f"Optional[{result}]"
                return result

        # Check if it's a known enum
        for enum in self.enums:
            if go_type == enum.name:
                if is_pointer or is_optional:
                    return f"Optional[{enum.name}]"
                return enum.name

        # Check if it's a struct we're defining
        struct_names = [s.name for s in self.structs]
        if go_type in struct_names:
            if is_pointer or is_optional:
                return f"Optional[{go_type}]"
            return go_type

        # Use type map
        python_type = self.TYPE_MAP.get(go_type, go_type)

        if is_pointer or is_optional:
            return f"Optional[{python_type}]"

        return python_type

    def generate_pydantic(self) -> str:
        """Generate Pydantic models from parsed structs"""
        lines = [
            _SPDX_HEADER,
            '"""',
            "Auto-generated Pydantic models from v1beta1 DGDR Go types.",
            "",
            "Generated by: deploy/operator/api/scripts/generate_pydantic_from_go.py",
            "Source: deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go",
            "",
            "DO NOT EDIT MANUALLY - regenerate using the script.",
            '"""',
            "",
            "from enum import Enum",
            "from typing import Any, Dict, List, Optional",
            "",
            "from pydantic import BaseModel, Field, model_validator",
            "",
        ]

        # Emit import statements for overridden types, grouped by module
        import_groups: dict[str, list[str]] = {}
        for go_name, (mod, py_name) in _IMPORT_OVERRIDES.items():
            # Only emit if the type actually appears in the parsed content
            in_enums = any(e.name == go_name for e in self.enums)
            in_structs = any(s.name == go_name for s in self.structs)
            # Always emit if it's a known planner type (may appear as field type)
            if (
                in_enums
                or in_structs
                or go_name in ("PlannerPreDeploymentSweepMode", "PlannerConfig")
            ):
                import_groups.setdefault(mod, []).append(py_name)

        for mod in sorted(import_groups):
            names = sorted(import_groups[mod])
            lines.append(
                "# Import canonical planner types – do NOT redefine them here."
            )
            lines.append(f"from {mod} import (  # noqa: F401 (re-exported)")
            for n in names:
                lines.append(f"    {n},")
            lines.append(")")
            lines.append("")

        # Generate enums first (skip ones that are imported)
        for enum in self.enums:
            if enum.name in _IMPORT_OVERRIDES:
                continue  # imported above
            lines.append("")
            if enum.comment:
                lines.append(f"# {enum.comment}")
            lines.append(f"class {enum.name}(str, Enum):")
            if not enum.values:
                lines.append("    pass")
            else:
                for const_name, const_value in enum.values:
                    lines.append(f'    {const_name} = "{const_value}"')

        # Generate struct models
        for struct in self.structs:
            lines.append("")
            lines.append("")
            lines.append(f"class {struct.name}(BaseModel):")

            # Add docstring — prefer an explicit override, fall back to Go comment
            docstring = _STRUCT_DOCSTRINGS.get(struct.name) or struct.comment
            if docstring:
                if "\n" in docstring:
                    lines.append(f'    """{docstring}"""')
                else:
                    lines.append(f'    """{docstring}"""')
                lines.append("")

            if not struct.fields:
                lines.append("    pass")
                continue

            # Generate fields
            for go_field in struct.fields:
                # Optional-wrapping rules:
                # - Non-pointer + default: NOT Optional. The API server (kubebuilder:default)
                #   or the Python default ensures the field always has a value.
                # - Pointer + default: keep Optional. The pointer can still be nil in the
                #   API object (e.g. +python-default is Python-layer only); callers may
                #   explicitly pass None.
                # - No default: follow is_optional (omitempty → Optional).
                effective_optional = go_field.is_optional and (
                    go_field.is_pointer or go_field.default is None
                )
                override_key = (struct.name, go_field.name)
                if override_key in _FIELD_TYPE_OVERRIDES:
                    python_type = _FIELD_TYPE_OVERRIDES[override_key]
                else:
                    python_type = self._go_type_to_python(
                        go_field.go_type, go_field.is_pointer, effective_optional
                    )

                field_def = f"    {go_field.name}: {python_type}"

                if (
                    go_field.comment
                    or effective_optional
                    or go_field.default is not None
                ):
                    field_args = []
                    if go_field.default is not None:
                        # Emit the default from kubebuilder annotation
                        raw = go_field.default
                        # Strip surrounding Go-style double quotes if present
                        if raw.startswith('"') and raw.endswith('"'):
                            raw = raw[1:-1]
                        # Try to cast to int/float/bool; otherwise keep as string
                        if raw.lower() == "true":
                            field_args.append("default=True")
                        elif raw.lower() == "false":
                            field_args.append("default=False")
                        else:
                            try:
                                int(raw)
                                field_args.append(f"default={raw}")
                            except ValueError:
                                try:
                                    float(raw)
                                    field_args.append(f"default={raw}")
                                except ValueError:
                                    field_args.append(f'default="{raw}"')
                    elif effective_optional:
                        field_args.append("default=None")
                    if go_field.comment:
                        comment_escaped = go_field.comment.replace('"', '\\"')
                        field_args.append(f'description="{comment_escaped}"')

                    field_def += f' = Field({", ".join(field_args)})'

                lines.append(field_def)

            # Append any struct-specific extras (validators, etc.)
            extra = _STRUCT_EXTRAS.get(struct.name)
            if extra:
                lines.append("")
                for extra_line in extra.splitlines():
                    lines.append(extra_line)

        return "\n".join(lines)


def main():
    script_dir = Path(__file__).parent.resolve()
    repo_root = _resolve_repo_root(script_dir)

    parser = argparse.ArgumentParser(
        description="Convert Go DGDR types to Python Pydantic models"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "v1beta1" / "dynamographdeploymentrequest_types.go",
        help="Input Go file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "components"
        / "src"
        / "dynamo"
        / "profiler"
        / "utils"
        / "dgdr_v1beta1_types.py",
        help="Output Python file path",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing Go types from: {args.input}")
    converter = GoToPydanticConverter()
    converter.parse_go_file(args.input)

    print(f"Found {len(converter.enums)} enums and {len(converter.structs)} structs")

    pydantic_code = converter.generate_pydantic()

    args.output.write_text(pydantic_code + "\n")
    print(f"Generated Pydantic models at: {args.output}")

    # Format with black to match the project linter style
    try:
        subprocess.run(
            ["black", "--line-length=88", "--quiet", str(args.output)],
            check=True,
        )
        print(f"Formatted with black: {args.output}")
    except FileNotFoundError:
        print("Warning: black not found; output may not match linter formatting")
    except subprocess.CalledProcessError as e:
        print(f"Warning: black formatting failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())

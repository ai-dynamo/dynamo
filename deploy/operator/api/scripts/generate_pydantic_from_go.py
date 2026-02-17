#!/usr/bin/env python3
"""
Script to convert Go struct definitions from v1beta1 DGDR types to Python Pydantic models.

This script parses the Go file containing Kubernetes CRD type definitions and generates
corresponding Pydantic models that can be used in Python code (e.g., in the profiler).

Usage:
    python generate_pydantic_from_go.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class GoField:
    """Represents a field in a Go struct"""

    name: str
    go_type: str
    json_tag: str
    comment: str
    is_optional: bool
    is_pointer: bool


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
    TYPE_MAP = {
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

            # Extract comment
            lines_before = content[: match.start()].split("\n")
            comment_lines = []
            for line in reversed(lines_before[-10:]):  # Look at last 10 lines
                if line.strip().startswith("//") and "kubebuilder" not in line:
                    comment_lines.insert(0, line.strip("/ ").strip())
                elif line.strip() and not line.strip().startswith("//"):
                    break

            # Parse enum values from kubebuilder annotation
            enum_values = [v.strip() for v in enum_values_str.split(";")]

            # Try to find const block with actual values
            const_pattern = rf'const\s+\(\s*\n((?:\s*{enum_name}\w+\s+{enum_name}\s+=\s+"[^"]+"\s*\n)+)\s*\)'
            const_match = re.search(const_pattern, content)

            values = []
            if const_match:
                # Extract individual const definitions
                const_defs = const_match.group(1)
                const_pattern_inner = rf'{enum_name}(\w+)\s+{enum_name}\s+=\s+"([^"]+)"'
                for const_match_inner in re.finditer(const_pattern_inner, const_defs):
                    const_name = const_match_inner.group(1)
                    const_value = const_match_inner.group(2)
                    values.append((const_name, const_value))
            else:
                # Use values from kubebuilder annotation
                values = [(v.upper().replace("-", "_"), v) for v in enum_values]

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
        # Pattern to match struct definitions with comments
        # Need to handle nested braces properly
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

            # Skip if this is a List type
            if struct_name.endswith("List"):
                continue

            # Extract comment from lines before struct
            lines_before = content[: match.start()].split("\n")
            comment_lines = []
            for line in reversed(
                lines_before[-20:]
            ):  # Look at more lines for struct comments
                line = line.strip()
                if (
                    line.startswith("//")
                    and "kubebuilder" not in line
                    and "EDIT THIS FILE" not in line
                ):
                    # Remove '//' and clean up
                    comment_lines.insert(0, line.lstrip("/ ").strip())
                elif line and not line.startswith("//"):
                    break

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

        # Split by newlines and process each line
        lines = struct_body.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments outside of field definitions
            if not line or (
                line.startswith("//")
                and i + 1 < len(lines)
                and not lines[i + 1].strip().startswith("//")
            ):
                i += 1
                continue

            # Collect multi-line comments
            comment_lines = []
            while line.startswith("//"):
                if "kubebuilder" not in line and "+optional" not in line.lower():
                    comment_lines.append(line.lstrip("/ ").strip())
                i += 1
                if i >= len(lines):
                    break
                line = lines[i].strip()

            # Now line should be a field definition
            # Pattern: FieldName type `json:"jsonName,omitempty"`
            field_pattern = r'(\w+)\s+([\w\.\*\[\]]+)\s+`json:"([^"]+)"`'
            match = re.match(field_pattern, line)

            if match:
                field_type = match.group(2)
                json_tag_full = match.group(3)

                # Parse json tag
                json_parts = json_tag_full.split(",")
                json_name = json_parts[0]
                is_optional = "omitempty" in json_parts or ",inline" in json_tag_full

                # Check if pointer type
                is_pointer = field_type.startswith("*")
                if is_pointer:
                    field_type = field_type[1:]  # Remove *

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
            # Arrays are optional if marked as omitempty
            if is_optional:
                return f"Optional[{result}]"
            return result

        # Handle map types
        if go_type.startswith("map["):
            # Extract key and value types
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
                # Enums can be optional too
                if is_pointer or is_optional:
                    return f"Optional[{enum.name}]"
                return enum.name

        # Check if it's a struct we're defining
        struct_names = [s.name for s in self.structs]
        if go_type in struct_names:
            # Nested structs are optional if pointer or omitempty
            if is_pointer or is_optional:
                return f"Optional[{go_type}]"
            return go_type

        # Use type map
        python_type = self.TYPE_MAP.get(go_type, go_type)

        # Add Optional wrapper if pointer or optional
        if is_pointer or is_optional:
            return f"Optional[{python_type}]"

        return python_type

    def generate_pydantic(self) -> str:
        """Generate Pydantic models from parsed structs"""
        lines = [
            '"""',
            "Auto-generated Pydantic models from v1beta1 DGDR Go types.",
            "",
            "Generated by: deploy/operator/api/scripts/generate_pydantic_from_go.py",
            "Source: deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go",
            "",
            "DO NOT EDIT MANUALLY - regenerate using the script.",
            '"""',
            "",
            "from typing import Optional, List, Dict, Any",
            "from pydantic import BaseModel, Field",
            "from enum import Enum",
            "",
        ]

        # Generate enums first
        for enum in self.enums:
            lines.append("")
            if enum.comment:
                lines.append(f"# {enum.comment}")
            lines.append(f"class {enum.name}(str, Enum):")
            if not enum.values:
                lines.append("    pass")
            else:
                for const_name, const_value in enum.values:
                    # Handle Python reserved keywords
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
                    lines.append(f'    {const_name} = "{const_value}"')

        # Generate struct models
        for struct in self.structs:
            lines.append("")
            lines.append("")
            lines.append(f"class {struct.name}(BaseModel):")

            # Add docstring
            if struct.comment:
                lines.append(f'    """{struct.comment}"""')
                lines.append("")

            if not struct.fields:
                lines.append("    pass")
                continue

            # Generate fields
            for field in struct.fields:
                python_type = self._go_type_to_python(
                    field.go_type, field.is_pointer, field.is_optional
                )

                # Build field definition
                field_def = f"    {field.name}: {python_type}"

                # Add Field() with description if there's a comment
                if field.comment or field.is_optional:
                    field_args = []
                    if field.is_optional:
                        field_args.append("default=None")
                    if field.comment:
                        # Escape quotes in comment
                        comment_escaped = field.comment.replace('"', '\\"')
                        field_args.append(f'description="{comment_escaped}"')

                    field_def += f' = Field({", ".join(field_args)})'

                lines.append(field_def)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Go DGDR types to Python Pydantic models"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent
        / "v1beta1"
        / "dynamographdeploymentrequest_types.go",
        help="Input Go file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent.parent
        / "components"
        / "src"
        / "dynamo"
        / "profiler"
        / "utils"
        / "dgdr_v1beta1_types.py",
        help="Output Python file path",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Parse and convert
    print(f"Parsing Go types from: {args.input}")
    converter = GoToPydanticConverter()
    converter.parse_go_file(args.input)

    print(f"Found {len(converter.enums)} enums and {len(converter.structs)} structs")

    # Generate Pydantic code
    pydantic_code = converter.generate_pydantic()

    # Write output
    args.output.write_text(pydantic_code)
    print(f"Generated Pydantic models at: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())

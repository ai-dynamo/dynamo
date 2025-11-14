#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate DynamoGraphDeployment YAML files against the CRD schema.

This script validates deploy.yaml files without requiring kubectl or a Kubernetes cluster.
It dynamically uses the OpenAPI v3 schema from the CRD to validate all structure,
required fields, types, enums, and constraints.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("Error: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)


class DeploymentValidator:
    """Validates DynamoGraphDeployment manifests against CRD schema."""

    def __init__(self, crd_path: Path):
        """Initialize validator with CRD schema."""
        self.crd_path = crd_path
        self.schema = self._load_crd_schema()
        self.validator = Draft7Validator(self.schema)

    def _load_crd_schema(self) -> Dict[str, Any]:
        """Load and extract OpenAPI schema from CRD."""
        try:
            with open(self.crd_path) as f:
                crd = yaml.safe_load(f)

            # Extract the OpenAPI v3 schema for v1alpha1
            for version in crd.get("spec", {}).get("versions", []):
                if version.get("name") == "v1alpha1":
                    schema = version.get("schema", {}).get("openAPIV3Schema", {})
                    if not schema:
                        raise ValueError("OpenAPI schema is empty")
                    return schema

            raise ValueError("Could not find v1alpha1 schema in CRD")
        except Exception as e:
            print(f"Error loading CRD schema from {self.crd_path}: {e}")
            sys.exit(1)

    def validate_file(self, yaml_file: Path) -> Tuple[bool, List[str]]:
        """
        Validate a single YAML file.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            with open(yaml_file) as f:
                content = f.read()
                docs = list(yaml.safe_load_all(content))
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]

        # Find DynamoGraphDeployment documents
        dgd_docs = [
            (i, doc)
            for i, doc in enumerate(docs)
            if doc and doc.get("kind") == "DynamoGraphDeployment"
        ]

        if not dgd_docs:
            return True, []  # No DGD resources, skip validation

        # Validate each DynamoGraphDeployment
        errors = []
        for doc_idx, doc in dgd_docs:
            self._file_content = content
            errors.extend(self._validate_document(doc))

        return len(errors) == 0, errors

    def _validate_document(self, doc: Dict[str, Any]) -> List[str]:
        """Validate a single DynamoGraphDeployment document."""
        errors = []

        # JSON Schema validation
        for error in sorted(self.validator.iter_errors(doc), key=lambda e: e.path):
            errors.append(self._format_schema_error(error))

        # Additional structural linting (catch indentation issues)
        errors.extend(self._lint_structure(doc, [], []))

        return errors

    def _format_schema_error(self, error) -> str:
        """Format a JSON schema validation error into a readable message."""
        path = self._format_path(error.path)

        # Try to find line number for this error
        line_num = self._find_line_for_error(error)
        line_prefix = f"Line {line_num}: " if line_num else ""

        # Create specific error messages based on validator type
        if error.validator == "enum":
            allowed_str = ", ".join(f"'{v}'" for v in error.validator_value)
            msg = f"{path}: value '{error.instance}' must be one of: {allowed_str}"
        elif error.validator == "type":
            msg = f"{path}: expected type '{error.validator_value}', got '{type(error.instance).__name__}'"
        elif error.validator == "required":
            msg = (
                f"{path}: missing required field(s): {', '.join(error.validator_value)}"
            )
        elif error.validator == "maxProperties":
            msg = f"{path}: too many properties (max: {error.validator_value})"
        elif error.validator == "maxItems":
            msg = f"{path}: too many items (max: {error.validator_value})"
        elif error.validator == "minProperties":
            msg = f"{path}: too few properties (min: {error.validator_value})"
        elif error.validator == "minItems":
            msg = f"{path}: too few items (min: {error.validator_value})"
        elif error.validator == "pattern":
            msg = f"{path}: value does not match pattern '{error.validator_value}'"
        elif error.validator == "minimum":
            msg = f"{path}: value must be >= {error.validator_value}"
        elif error.validator == "maximum":
            msg = f"{path}: value must be <= {error.validator_value}"
        else:
            msg = f"{path}: {error.message}"

        return f"{line_prefix}{msg}"

    def _lint_structure(
        self, obj: Any, schema_path: List[Any], data_path: List[Any]
    ) -> List[str]:
        """
        Recursively validate that all properties in the document are defined in the schema.
        This catches indentation errors where properties appear at the wrong level.
        """
        if not isinstance(obj, dict):
            return []

        errors = []
        schema_at_path = self._navigate_schema(schema_path)

        if not schema_at_path or not isinstance(schema_at_path, dict):
            return []

        # Get allowed properties from schema
        allowed = set(schema_at_path.get("properties", {}).keys())

        # Check each property
        for key, value in obj.items():
            if allowed and key not in allowed:
                # Property not defined in schema - likely indentation error
                path_str = self._format_path(data_path + [key])
                errors.append(
                    f"{path_str}: unexpected property '{key}' "
                    f"(not defined in schema - check indentation)"
                )

            # Recursively check nested structures
            if key in allowed or not allowed:
                if isinstance(value, dict):
                    errors.extend(
                        self._lint_structure(
                            value, schema_path + [key], data_path + [key]
                        )
                    )
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, dict):
                            errors.extend(
                                self._lint_structure(
                                    item,
                                    schema_path + [key, idx],
                                    data_path + [key, idx],
                                )
                            )

        return errors

    def _navigate_schema(self, path: List[Any]) -> Optional[Dict[str, Any]]:
        """Navigate to a specific path in the schema."""
        current = self.schema

        for step in path:
            if not isinstance(current, dict):
                return None

            if isinstance(step, int):
                # Array index - use items schema
                current = current.get("items")
            else:
                # Object property
                properties = current.get("properties", {})
                if step in properties:
                    current = properties[step]
                elif isinstance(current.get("additionalProperties"), dict):
                    current = current["additionalProperties"]
                else:
                    return None

            if current is None:
                return None

        return current

    def _find_line(self, property_name: str) -> Optional[int]:
        """Find line number for a property name in the YAML file."""
        if not hasattr(self, "_file_content"):
            return None

        search_pattern = f"{property_name}:"
        for i, line in enumerate(self._file_content.split("\n")):
            if search_pattern in line:
                return i + 1

        return None

    def _find_line_for_error(self, error) -> Optional[int]:
        """Find line number for a JSON schema validation error."""
        if not hasattr(self, "_file_content") or not error.path:
            return None

        # Get the last property name in the path (where the error occurred)
        path_list = list(error.path)
        for item in reversed(path_list):
            if not isinstance(item, int):
                return self._find_line(item)

        return None

    def _format_path(self, path) -> str:
        """Format a path (list or deque) to a readable string."""
        if not path:
            return "root"

        parts = []
        for item in path:
            if isinstance(item, int):
                parts.append(f"[{item}]")
            else:
                parts.append(f".{item}" if parts else str(item))

        return "".join(parts)


def has_dgd_resource(yaml_file: Path) -> bool:
    """Check if a YAML file contains DynamoGraphDeployment resources."""
    try:
        with open(yaml_file) as f:
            for doc in yaml.safe_load_all(f):
                if doc and doc.get("kind") == "DynamoGraphDeployment":
                    return True
        return False
    except Exception:
        return False


def find_deploy_files(root_dir: Path, limit_dirs: List[str] = None) -> List[Path]:
    """
    Find all YAML files containing DynamoGraphDeployment resources.

    Uses git ls-files to respect .gitignore, falling back to glob if git is unavailable.

    Args:
        root_dir: Root directory to search from
        limit_dirs: Optional list of subdirectories to limit search to (for speed)
    """
    dgd_files = []

    # If specific directories requested, search only those
    if limit_dirs:
        for dir_name in limit_dirs:
            search_dir = root_dir / dir_name
            if not search_dir.exists():
                continue
            for ext in ["*.yaml", "*.yml"]:
                for yaml_file in search_dir.glob(f"**/{ext}"):
                    if has_dgd_resource(yaml_file):
                        dgd_files.append(yaml_file)
        return sorted(dgd_files)

    # Otherwise, search entire repo using git
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.yaml", "*.yml"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        yaml_files = [
            root_dir / Path(f) for f in result.stdout.strip().split("\n") if f
        ]
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        # Fallback: search entire repo (excluding common directories)
        yaml_files = []
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
        for ext in ["*.yaml", "*.yml"]:
            for yaml_file in root_dir.glob(f"**/{ext}"):
                if not any(part in exclude_dirs for part in yaml_file.parts):
                    yaml_files.append(yaml_file)

    # Filter to only files containing DynamoGraphDeployment
    for yaml_file in yaml_files:
        if yaml_file.exists() and has_dgd_resource(yaml_file):
            dgd_files.append(yaml_file)

    return sorted(dgd_files)


def main():
    parser = argparse.ArgumentParser(
        description="Validate DynamoGraphDeployment YAML files against CRD schema"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="YAML files to validate (default: find all YAML files with DynamoGraphDeployment resources, respecting .gitignore)",
    )
    parser.add_argument(
        "--crd", type=Path, help="Path to CRD schema file (default: auto-detect)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (treat warnings as errors)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Limit search to specific directories (e.g., --dirs recipes examples tests)",
    )

    args = parser.parse_args()

    # Determine workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent

    # Find CRD schema
    crd_path = args.crd or (
        workspace_root
        / "deploy/cloud/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml"
    )

    if not crd_path.exists():
        print(f"Error: CRD schema not found at {crd_path}")
        print("Specify the CRD path with --crd")
        return 1

    # Find files to validate
    if args.files:
        files_to_validate = [Path(f) for f in args.files]
    else:
        files_to_validate = find_deploy_files(workspace_root, limit_dirs=args.dirs)
        if not files_to_validate:
            print("No YAML files with DynamoGraphDeployment resources found")
            return 0

    if args.verbose:
        print(f"Using CRD schema: {crd_path}")
        print(f"Validating {len(files_to_validate)} file(s)...")
        print()

    # Validate files
    validator = DeploymentValidator(crd_path)
    all_valid = True
    total_errors = 0

    for yaml_file in files_to_validate:
        if not yaml_file.exists():
            print(f"❌ {yaml_file}: File not found")
            all_valid = False
            continue

        is_valid, errors = validator.validate_file(yaml_file)

        if is_valid:
            if args.verbose:
                print(f"✅ {yaml_file}: Valid")
        else:
            print(f"❌ {yaml_file}: {len(errors)} error(s)")
            for error in errors:
                print(f"   - {error}")
            print()
            all_valid = False
            total_errors += len(errors)

    # Summary
    print()
    if all_valid:
        print(f"✅ All {len(files_to_validate)} file(s) are valid!")
        return 0
    else:
        print(f"❌ Validation failed with {total_errors} error(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

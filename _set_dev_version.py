#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This script is used to set the development version for the dynamo project.

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tomlkit


def find_all_pyproject_toml_files():
    script_dir = Path(__file__).parent
    pyproject_files = []
    for pyproject_file in script_dir.rglob("pyproject.toml"):
        pyproject_files.append(pyproject_file)
    return pyproject_files


def extract_pyproject_toml_info(files, param):
    results = []
    for path in files:
        try:
            with open(path, "r") as f:
                project = tomlkit.parse(f.read()).get("project", {})
            results.append(project.get(param))
            # results.append({p: project.get(p) for p in params})
        except Exception as e:
            results.append({"error": str(e)})
    return results


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def validate_python_version(version: str) -> bool:
    # https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    return (
        re.match(
            r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
            version,
        )
        is not None
    )


def define_dev_version(current_version: str):
    if not validate_python_version(current_version):
        raise ValueError(f"Invalid version format: {current_version}")
    if not os.environ.get("CI_PIPELINE_ID"):
        raise ValueError(
            "CI_PIPELINE_ID is not set, please set it in the environment variables"
        )
    if not get_git_hash():
        raise ValueError("Git hash is not set")
    version = re.match(r"(\d+\.\d+\.\d+)", current_version).group(1)
    pipeline_id = (
        ".dev" + os.environ.get("CI_PIPELINE_ID")
        if os.environ.get("CI_PIPELINE_ID")
        else ""
    )
    if not validate_python_version(f"{version}{pipeline_id}"):
        raise ValueError(f"Invalid version format: {version}")

    hash = "+" + get_git_hash() if get_git_hash() else ""
    version = f"{version}{pipeline_id}{hash}"
    return version


def align_versions(projects, version, dependencies):
    updated_deps = []
    for dep_group in dependencies:
        new_group = []
        for dep in dep_group:
            matched = False
            for proj in projects:
                if (
                    dep.startswith(f"{proj}==")
                    or dep.startswith(f"{proj}>=")
                    or dep.startswith(f"{proj}<=")
                    or dep.startswith(f"{proj}")
                ):
                    new_group.append(f"{proj}=={version}")
                    matched = True
                    break
            if not matched:
                new_group.append(dep)
        updated_deps.append(new_group)
    return updated_deps


def update_pyproject_toml_version(files, version):
    for file in files:
        with open(file, "r") as f:
            doc = tomlkit.parse(f.read())
        doc["project"]["version"] = version
        with open(file, "w") as f:
            f.write(tomlkit.dumps(doc))


def update_pyproject_toml_dependencies(files, dependencies):
    for file, dep_group in zip(files, dependencies):
        with open(file, "r") as f:
            doc = tomlkit.parse(f.read())
        doc["project"]["dependencies"] = dep_group
        with open(file, "w") as f:
            f.write(tomlkit.dumps(doc))


def display_version_summary(
    pyproject_files,
    package_names,
    current_versions,
    current_dependencies,
    updated_dependencies,
    dev_version,
):
    """Display a comprehensive summary of version management operations."""
    # Display summary in a structured format
    print("\n" + "=" * 80)
    print("VERSION MANAGEMENT SUMMARY")
    print("=" * 80)

    print(f"\n📁 Working Directory: {Path(__file__).parent}")
    print(f"🔍 Found {len(pyproject_files)} pyproject.toml file(s)")
    print(f"🚀 New Development Version: {dev_version}")

    # Display project versions table
    print("\n📊 PROJECT VERSIONS")
    print("-" * 80)
    print(f"{'Project Name':<25} {'Current Version':<20} {'New Version':<20}")
    print("-" * 80)
    for i, (file_path, name, old_ver) in enumerate(
        zip(pyproject_files, package_names, current_versions)
    ):
        if (
            name
            and old_ver
            and not isinstance(name, dict)
            and not isinstance(old_ver, dict)
        ):
            print(f"{name:<25} {old_ver:<20} {dev_version:<20}")
        else:
            status = (
                "❌ Error"
                if isinstance(name, dict) or isinstance(old_ver, dict)
                else "⚠️  Missing"
            )
            print(f"{file_path:<25} {status:<20} {dev_version:<20}")

    # Display dependencies alignment
    print("\n🔗 DEPENDENCIES ALIGNMENT")
    print("-" * 80)
    for i, (file_path, old_deps, new_deps) in enumerate(
        zip(pyproject_files, current_dependencies, updated_dependencies)
    ):
        if old_deps and not isinstance(old_deps, dict):
            print(f"\n📁 {file_path}:")
            print("  Before → After:")
            for old_dep, new_dep in zip(old_deps, new_deps):
                if old_dep != new_dep:
                    print(f"    {old_dep:<30} → {new_dep}")
                else:
                    print(f"    {old_dep:<30} → (unchanged)")
        elif isinstance(old_deps, dict):
            print(f"\n📁 {file_path}: ❌ Error reading dependencies")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Versions will be updated to: {dev_version}")
    print(f"✅ Dependencies will be aligned across {len(pyproject_files)} projects")
    print("=" * 80)


def main():
    pyproject_toml_files = find_all_pyproject_toml_files()
    current_versions = extract_pyproject_toml_info(pyproject_toml_files, "version")
    current_version = current_versions[0]
    current_package_names = extract_pyproject_toml_info(pyproject_toml_files, "name")
    current_dependencies = extract_pyproject_toml_info(
        pyproject_toml_files, "dependencies"
    )
    dev_version = define_dev_version(current_version)
    updated_dependencies = align_versions(
        current_package_names, dev_version, current_dependencies
    )
    display_version_summary(
        pyproject_toml_files,
        current_package_names,
        current_versions,
        current_dependencies,
        updated_dependencies,
        dev_version,
    )

    print("\n🔄 UPDATING FILES...")
    print("-" * 40)

    # Update versions
    print("📝 Updating project versions...")
    update_pyproject_toml_version(pyproject_toml_files, dev_version)
    print("✅ Project versions updated successfully")

    # Update dependencies
    print("🔗 Updating dependencies...")
    update_pyproject_toml_dependencies(pyproject_toml_files, updated_dependencies)
    print("✅ Dependencies updated successfully")

    print("\n🎉 All updates completed successfully!")
    print(f"📊 {len(pyproject_toml_files)} pyproject.toml files processed")
    print("=" * 80)


if __name__ == "__main__":
    main()

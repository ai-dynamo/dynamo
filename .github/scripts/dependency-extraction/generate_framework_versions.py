#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Generate FRAMEWORK_VERSIONS.md dynamically from dependency extraction data.

This script reads the latest dependency CSV and generates a markdown document
showing critical framework versions, base images, and configurations.

Usage:
    python3 generate_framework_versions.py \\
        --csv .github/reports/dependency_versions_latest.csv \\
        --output FRAMEWORK_VERSIONS.md
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class FrameworkVersionsGenerator:
    """Generates FRAMEWORK_VERSIONS.md from dependency extraction CSV."""

    def __init__(self, csv_path: Path, release_csv_path: Path = None):
        """
        Initialize with path to dependency CSV.
        
        Args:
            csv_path: Path to latest dependency CSV
            release_csv_path: Optional path to most recent release CSV
        """
        self.csv_path = csv_path
        self.release_csv_path = release_csv_path
        
        # Load latest dependencies
        self.dependencies = self._load_csv(csv_path)
        self.critical_deps = self._filter_critical(self.dependencies)
        self.base_images = self._filter_base_images(self.dependencies)
        
        # Load release dependencies if available
        self.release_dependencies = None
        self.release_version = None
        if release_csv_path and release_csv_path.exists():
            self.release_dependencies = self._load_csv(release_csv_path)
            # Extract version from filename: dependency_versions_vX.Y.Z.csv
            import re
            match = re.search(r'v(\d+\.\d+\.\d+)', str(release_csv_path))
            if match:
                self.release_version = match.group(1)

    def _load_csv(self, csv_path: Path) -> List[Dict[str, str]]:
        """Load dependencies from CSV."""
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _filter_critical(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter for critical dependencies only."""
        return [d for d in dependencies if d["Critical"] == "Yes"]

    def _filter_base_images(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter for base/runtime images."""
        return [
            d
            for d in dependencies
            if d["Category"] in ("Base Image", "Runtime Image")
        ]
    
    def _get_release_version(self, dep_name: str, component: str) -> str:
        """Get version from release CSV for a specific dependency."""
        if not self.release_dependencies:
            return "N/A"
        
        for dep in self.release_dependencies:
            if (dep["Dependency Name"] == dep_name and 
                dep["Component"] == component):
                return dep["Version"]
        return "N/A"

    def generate(self) -> str:
        """Generate the markdown content."""
        lines = []

        # Header
        lines.extend(self._generate_header())

        # Core Framework Dependencies
        lines.extend(self._generate_core_frameworks())

        # Base Images
        lines.extend(self._generate_base_images())

        # Framework-Specific Configurations
        lines.extend(self._generate_framework_configs())

        # Dependency Management
        lines.extend(self._generate_dependency_management())

        # Notes
        lines.extend(self._generate_notes())

        # Footer with links
        lines.extend(self._generate_footer())

        return "\n".join(lines)

    def _generate_header(self) -> List[str]:
        """Generate document header."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        header_lines = [
            "<!--",
            "SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "SPDX-License-Identifier: Apache-2.0",
            "",
            "Licensed under the Apache License, Version 2.0 (the \"License\");",
            "you may not use this file except in compliance with the License.",
            "You may obtain a copy of the License at",
            "",
            "http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software",
            "distributed under the License is distributed on an \"AS IS\" BASIS,",
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "See the License for the specific language governing permissions and",
            "limitations under the License.",
            "-->",
            "",
            "# Dynamo Framework & Dependency Versions",
            "",
            f"> **⚠️ AUTO-GENERATED** - Last updated: {timestamp}",
            "> ",
            "> This document is automatically generated from [dependency extraction](.github/reports/dependency_versions_latest.csv).",
            "> To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`",
            "",
        ]
        
        if self.release_version:
            header_lines.extend([
                f"**Comparison:** Latest (main) vs Release v{self.release_version}",
                "",
                "This document shows critical framework versions from both:",
                "- **Latest (main branch)**: Current development versions",
                f"- **Release v{self.release_version}**: Last stable release",
                "",
            ])
        else:
            header_lines.extend([
                "This document tracks the major dependencies and critical versions used in the NVIDIA Dynamo project.",
                "",
            ])
        
        return header_lines

    def _generate_core_frameworks(self) -> List[str]:
        """Generate core framework dependencies section."""
        lines = ["## Core Framework Dependencies", ""]

        # Group by component
        by_component = defaultdict(list)
        for dep in self.critical_deps:
            # Skip base images (separate section)
            if dep["Category"] in ("Base Image", "Runtime Image"):
                continue
            by_component[dep["Component"]].append(dep)

        # Core frameworks to highlight
        framework_priority = [
            ("vllm", "vLLM", "High-throughput LLM serving engine"),
            (
                "trtllm",
                "TensorRT-LLM",
                "NVIDIA's optimized inference library for large language models",
            ),
            ("sglang", "SGLang", "Structured generation language for LLMs"),
        ]

        for comp_key, comp_name, description in framework_priority:
            lines.append(f"### {comp_name}")

            # Find the main framework dependency
            framework_dep = next(
                (
                    d
                    for d in by_component.get(comp_key, [])
                    if comp_key in d["Dependency Name"].lower()
                    or comp_name.lower().replace("-", "") in d["Dependency Name"].lower()
                ),
                None,
            )

            if framework_dep:
                latest_version = framework_dep['Version']
                release_version = self._get_release_version(
                    framework_dep['Dependency Name'], 
                    framework_dep['Component']
                )
                
                # Show both latest and release versions
                if self.release_version and release_version != "N/A":
                    lines.append(f"- **Latest (main)**: `{latest_version}`")
                    lines.append(f"- **Release (v{self.release_version})**: `{release_version}`")
                    if latest_version != release_version:
                        lines.append(f"  - ⚠️ _Version difference detected_")
                else:
                    lines.append(f"- **Version**: `{latest_version}`")
                
                lines.append(f"- **Description**: {description}")
                if framework_dep["Package Source URL"] != "N/A":
                    lines.append(
                        f"- **Source**: [{framework_dep['Package Source URL']}]({framework_dep['Package Source URL']})"
                    )
                lines.append(f"- **Component**: `{framework_dep['Component']}`")
            else:
                lines.append("- **Version**: See dependency reports")

            lines.append("")

        # Other critical dependencies
        other_critical = [
            d
            for d in self.critical_deps
            if d["Category"] not in ("Base Image", "Runtime Image")
            and d["Component"] not in ("vllm", "trtllm", "sglang")
        ]

        if other_critical:
            lines.append("### Additional Critical Dependencies")
            lines.append("")
            for dep in sorted(
                other_critical, key=lambda x: x["Dependency Name"].lower()
            ):
                lines.append(f"#### {dep['Dependency Name']}")
                lines.append(f"- **Version**: `{dep['Version']}`")
                lines.append(f"- **Category**: {dep['Category']}")
                lines.append(f"- **Component**: `{dep['Component']}`")
                if dep["Package Source URL"] != "N/A":
                    lines.append(f"- **Source**: {dep['Package Source URL']}")
                lines.append("")

        return lines

    def _generate_base_images(self) -> List[str]:
        """Generate base images section."""
        lines = ["## Base & Runtime Images", ""]

        # Group by component
        by_component = defaultdict(list)
        for img in self.base_images:
            by_component[img["Component"]].append(img)

        for component in sorted(by_component.keys()):
            images = by_component[component]
            lines.append(f"### {component.upper()} Container Images")
            lines.append("")

            for img in images:
                lines.append(f"#### {img['Dependency Name']}")
                lines.append(f"- **Tag**: `{img['Version']}`")
                lines.append(f"- **Category**: {img['Category']}")
                lines.append(f"- **Source File**: `{img['Source File']}`")
                if "cuda" in img["Dependency Name"].lower():
                    # Extract CUDA version from tag
                    version_match = img["Version"]
                    lines.append(f"- **CUDA Version**: Extracted from tag `{version_match}`")
                lines.append("")

        return lines

    def _generate_framework_configs(self) -> List[str]:
        """Generate framework-specific configurations."""
        lines = ["## Framework-Specific Configurations", ""]

        configs = {
            "vllm": {
                "title": "vLLM Configuration",
                "build_location": "`container/deps/vllm/install_vllm.sh`",
                "dockerfile": "`container/Dockerfile.vllm`",
            },
            "trtllm": {
                "title": "TensorRT-LLM Configuration",
                "build_location": "`container/build_trtllm_wheel.sh`",
                "dockerfile": "`container/Dockerfile.trtllm`",
            },
            "sglang": {
                "title": "SGLang Configuration",
                "build_location": "`container/Dockerfile.sglang`",
                "dockerfile": "`container/Dockerfile.sglang`",
            },
        }

        for comp_key, config in configs.items():
            lines.append(f"### {config['title']}")

            # Find critical deps for this component
            comp_deps = [d for d in self.critical_deps if d["Component"] == comp_key]

            if comp_deps:
                lines.append("**Critical Dependencies:**")
                for dep in comp_deps:
                    if dep["Category"] not in ("Base Image", "Runtime Image"):
                        lines.append(f"- {dep['Dependency Name']}: `{dep['Version']}`")
                lines.append("")

            lines.append(f"**Build Location**: {config['build_location']}")
            lines.append(f"**Dockerfile**: {config['dockerfile']}")
            lines.append("")

        return lines

    def _generate_dependency_management(self) -> List[str]:
        """Generate dependency management section."""
        return [
            "## Dependency Management",
            "",
            "### Automated Tracking",
            "Dependency versions are automatically extracted and tracked nightly.",
            "",
            "**Reports**:",
            "- Latest versions: [`.github/reports/dependency_versions_latest.csv`](.github/reports/dependency_versions_latest.csv)",
            "- Release snapshots: [`.github/reports/releases/`](.github/reports/releases/)",
            "- Documentation: [`.github/reports/README.md`](.github/reports/README.md)",
            "",
            "### Build Scripts",
            "- **Main Build Script**: `container/build.sh`",
            "- **vLLM Installation**: `container/deps/vllm/install_vllm.sh`",
            "- **TensorRT-LLM Wheel**: `container/build_trtllm_wheel.sh`",
            "- **NIXL Installation**: `container/deps/trtllm/install_nixl.sh`",
            "",
            "### Python Dependencies",
            "- **Core Requirements**: `container/deps/requirements.txt`",
            "- **Standard Requirements**: `container/deps/requirements.standard.txt`",
            "- **Test Requirements**: `container/deps/requirements.test.txt`",
            "",
        ]

    def _generate_notes(self) -> List[str]:
        """Generate notes section."""
        # Count total dependencies
        total_deps = len(self.dependencies)
        critical_count = len(self.critical_deps)
        nvidia_products = len([d for d in self.dependencies if d["NVIDIA Product"] == "Yes"])

        return [
            "## Statistics",
            "",
            f"- **Total Dependencies Tracked**: {total_deps}",
            f"- **Critical Dependencies**: {critical_count}",
            f"- **NVIDIA Products**: {nvidia_products}",
            "",
            "## Notes",
            "",
            "- Different frameworks may use slightly different CUDA versions for runtime images",
            "- NIXL and UCX are primarily used for distributed inference scenarios",
            "- FlashInfer integration varies by build type (source builds, ARM64)",
            "- Dependency versions are centrally managed through Docker build arguments and shell script variables",
            "- Version discrepancies across components are automatically detected and reported",
            "",
        ]

    def _generate_footer(self) -> List[str]:
        """Generate footer with links."""
        return [
            "## Container Documentation",
            "",
            "For detailed information about container builds and usage, see:",
            "- [Container README](container/README.md)",
            "- [Container Build Script](container/build.sh)",
            "- [Container Run Script](container/run.sh)",
            "",
            "## Related Documentation",
            "",
            "- [Support Matrix](docs/support_matrix.md) - Supported platforms and versions",
            "- [Dependency Extraction System](.github/scripts/dependency-extraction/README.md) - How dependencies are tracked",
            "- [Dependency Reports](.github/reports/README.md) - CSV structure and workflows",
            "",
            "---",
            "",
            "_This document is automatically generated. Do not edit manually._",
            "_To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`_",
        ]


def find_latest_release_csv(releases_dir: Path) -> Path:
    """Find the most recent release CSV by version number."""
    import re
    
    if not releases_dir.exists():
        return None
    
    release_files = list(releases_dir.glob("dependency_versions_v*.csv"))
    if not release_files:
        return None
    
    # Extract version numbers and sort
    versioned_files = []
    for f in release_files:
        match = re.search(r'v(\d+)\.(\d+)\.(\d+)', f.name)
        if match:
            major, minor, patch = map(int, match.groups())
            versioned_files.append(((major, minor, patch), f))
    
    if not versioned_files:
        return None
    
    # Sort by version (latest first)
    versioned_files.sort(reverse=True)
    return versioned_files[0][1]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate FRAMEWORK_VERSIONS.md from dependency CSV"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(".github/reports/dependency_versions_latest.csv"),
        help="Path to latest dependency CSV file",
    )
    parser.add_argument(
        "--release-csv",
        type=Path,
        default=None,
        help="Path to release dependency CSV file (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FRAMEWORK_VERSIONS.md"),
        help="Output markdown file path",
    )

    args = parser.parse_args()

    # Auto-detect latest release CSV if not specified
    release_csv = args.release_csv
    if not release_csv:
        releases_dir = Path(".github/reports/releases")
        release_csv = find_latest_release_csv(releases_dir)
        if release_csv:
            print(f"📸 Found latest release snapshot: {release_csv.name}")

    # Generate the document
    generator = FrameworkVersionsGenerator(args.csv, release_csv)
    content = generator.generate()

    # Write to output file
    args.output.write_text(content)

    print(f"✅ Generated {args.output}")
    print(f"📊 Total dependencies: {len(generator.dependencies)}")
    print(f"🔥 Critical dependencies: {len(generator.critical_deps)}")
    print(f"🐳 Base images: {len(generator.base_images)}")
    if generator.release_version:
        print(f"🎯 Comparing with release: v{generator.release_version}")


if __name__ == "__main__":
    main()


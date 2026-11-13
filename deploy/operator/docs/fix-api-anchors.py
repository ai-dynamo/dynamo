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

"""Post-process api-reference.md after crd-ref-docs generation.

crd-ref-docs generates anchors solely from type names, so types that exist in multiple
API versions get identical anchors (e.g. #dynamographdeploymentrequest). In standard
Markdown renderers the first occurrence wins. This script gives duplicate v1beta1 and
v1beta2 headings version-specific anchors and rewrites same-package links to them.
crd-ref-docs-config.yaml supplies package-aware links to v1beta1 types.

crd-ref-docs also renders links for some external dangerous types that are referenced
from the CRD but not emitted as sections. Strip those links so the published
reference does not contain dead anchors.

The DCD CRDs intentionally prune DGD-only fields from shared Go types, but
crd-ref-docs reads the Go types directly. Project those fields out of the standalone
DCD documentation so the generated reference matches the installed CRDs.
"""
import re
import sys
from pathlib import Path

TYPE_HEADING_RE = re.compile(
    r"^####\s+(?:(?:v1beta1|v1beta2)\s+)?(?P<name>\S+)\s*$",
    re.MULTILINE,
)
PACKAGE_HEADING_RE = re.compile(r"^##\s+(?P<name>\S+)\s*$", re.MULTILINE)
DCD_SPEC = "DynamoComponentDeploymentSpec"
DGD_ONLY_DCD_REFERENCES = {
    "DynamoComponentDeploymentSharedSpec",
    "ComponentRoleSpec",
    "ProviderOverride",
    "LPXConfig",
}
DCD_REFERENCE_RE = re.compile(r"^- \[DynamoComponentDeploymentSpec\]\(#[^)]+\)\s*$")


def prefix_duplicate_type_anchors(
    markdown: str, target_package: str, prefix: str
) -> str:
    """Give types duplicated from earlier API packages package-specific anchors."""
    package_matches = list(PACKAGE_HEADING_RE.finditer(markdown))
    target_index = next(
        (
            index
            for index, match in enumerate(package_matches)
            if match.group("name") == target_package
        ),
        None,
    )
    if target_index is None:
        return markdown

    target_match = package_matches[target_index]
    target_end = (
        package_matches[target_index + 1].start()
        if target_index + 1 < len(package_matches)
        else len(markdown)
    )
    target_part = markdown[target_match.start() : target_end]
    target_types = {
        match.group("name") for match in TYPE_HEADING_RE.finditer(target_part)
    }

    earlier_types = set()
    for index, package_match in enumerate(package_matches[:target_index]):
        if not package_match.group("name").startswith("nvidia.com/"):
            continue
        package_end = package_matches[index + 1].start()
        package_part = markdown[package_match.start() : package_end]
        earlier_types.update(
            match.group("name") for match in TYPE_HEADING_RE.finditer(package_part)
        )

    for type_name in sorted(target_types & earlier_types):
        target_part = re.sub(
            rf"^(####\s+)(?:{re.escape(prefix)}\s+)?{re.escape(type_name)}(\s*)$",
            rf"\1{prefix} {type_name}\2",
            target_part,
            flags=re.MULTILINE,
        )
        anchor = type_name.lower()
        target_part = target_part.replace(
            f"](#{anchor})",
            f"](#{prefix}-{anchor})",
        )

    return markdown[: target_match.start()] + target_part + markdown[target_end:]


def project_standalone_dcd_schema(markdown: str) -> str:
    """Remove DGD-only fields and references from standalone DCD docs."""
    output = []
    current_type = ""

    for line in markdown.splitlines(keepends=True):
        heading = TYPE_HEADING_RE.match(line)
        if heading:
            current_type = heading.group("name")

        if current_type == DCD_SPEC and line.startswith(
            ("| `providerOverride` ", "| `lpx` ")
        ):
            continue

        if current_type == DCD_SPEC and line.startswith("| `roles` "):
            columns = line.rstrip("\n").split(" | ")
            columns[0] = re.sub(
                r"_\[ComponentRoleSpec\]\(#[^)]+\) array_", "_object array_", columns[0]
            )
            old_note = (
                "Standalone DCD roles accept only `name` and `replicas`; "
                "`providerOverride` is a DGD-only provider context."
            )
            role_note = (
                "Standalone DCD roles support `name`, `replicas`, and `podTemplate`; "
                "`providerOverride` is a DGD-only provider context."
            )
            columns[1] = columns[1].replace(old_note, role_note)
            if role_note not in columns[1]:
                columns[1] += f" {role_note}"
            line = " | ".join(columns) + "\n"

        if current_type in DGD_ONLY_DCD_REFERENCES and DCD_REFERENCE_RE.match(line):
            continue

        output.append(line)

    return "".join(output)


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <api-reference.md>", file=sys.stderr)
    sys.exit(1)

path = Path(sys.argv[1])
content = path.read_text()

marker = "## nvidia.com/v1beta1"
idx = content.find(marker)
if idx == -1:
    print("Warning: v1beta1 section not found, skipping anchor fix", file=sys.stderr)
    sys.exit(0)

alpha_part = content[:idx]

# Keep the v1beta1 heading rewrite inside that package. The generated reference
# now contains later API packages with the same Go type names, and renaming their
# headings would recreate ambiguous or missing cross-version anchors.
next_package_idx = content.find("\n## ", idx + len(marker))
if next_package_idx == -1:
    beta_part = content[idx:]
    remaining_packages = ""
else:
    beta_part = content[idx:next_package_idx]
    remaining_packages = content[next_package_idx:]

# Match the package-qualified links in crd-ref-docs-config.yaml.
duplicate_types = [
    "DynamoGraphDeploymentRequest",
    "DynamoGraphDeploymentRequestNativeSpec",
    "DynamoGraphDeploymentRequestNativeStatus",
    "DynamoGraphDeploymentRequestSpec",
    "DynamoGraphDeploymentRequestStatus",
    "DynamoGraphDeploymentStatus",
    "DynamoGraphDeploymentSpec",
    "DynamoComponentDeploymentSharedSpec",
]

for t in duplicate_types:
    # Rename section headings: #### TypeName → #### v1beta1 TypeName
    beta_part = re.sub(
        r"(####\s+)" + re.escape(t) + r"(\s*$)",
        r"\1v1beta1 " + t + r"\2",
        beta_part,
        flags=re.MULTILINE,
    )

content = alpha_part + beta_part + remaining_packages
content = project_standalone_dcd_schema(content)

content = prefix_duplicate_type_anchors(content, "nvidia.com/v1beta2", "v1beta2")

external_types_without_sections = [
    "EndpointPickerConfig",
]

for t in external_types_without_sections:
    anchor = t.lower()
    content = content.replace(f"[{t}](#{anchor})", t)

path.write_text(content)
print(f"✅ Post-processed API reference in {path}")

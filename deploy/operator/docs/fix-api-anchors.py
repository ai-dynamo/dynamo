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

crd-ref-docs generates anchors solely from type names, so types that exist in both
API versions get identical anchors (e.g. #dynamographdeploymentrequest). In standard
Markdown renderers the first occurrence wins, meaning v1beta1 links resolve to the
v1alpha1 section. This script prepends "v1beta1 " to the affected headings in the
v1beta1 section. crd-ref-docs-config.yaml supplies package-aware links to them.

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

TYPE_HEADING_RE = re.compile(r"^####\s+(?:v1beta1\s+)?(?P<name>\S+)\s*$")
DCD_SPEC = "DynamoComponentDeploymentSpec"
DGD_ONLY_DCD_REFERENCES = {
    "DynamoComponentDeploymentSharedSpec",
    "ComponentRoleSpec",
    "ProviderOverride",
    "LPXConfig",
}
DCD_REFERENCE_RE = re.compile(r"^- \[DynamoComponentDeploymentSpec\]\(#[^)]+\)\s*$")


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
            if "Standalone DCD roles accept only" not in columns[1]:
                columns[1] += (
                    " Standalone DCD roles accept only `name` and `replicas`; "
                    "`providerOverride` is a DGD-only provider context."
                )
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
beta_part = content[idx:]

# Match the package-qualified links in crd-ref-docs-config.yaml.
duplicate_types = [
    "DynamoGraphDeploymentRequest",
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

content = alpha_part + beta_part
content = project_standalone_dcd_schema(content)

external_types_without_sections = [
    "EndpointPickerConfig",
]

for t in external_types_without_sections:
    anchor = t.lower()
    content = content.replace(f"[{t}](#{anchor})", t)

path.write_text(content)
print(f"✅ Post-processed API reference in {path}")

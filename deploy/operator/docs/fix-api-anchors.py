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
v1beta1 section and updates all intra-section links to match the new anchors.

crd-ref-docs also renders links for some external dangerous types that are referenced
from the CRD but not emitted as sections. Strip those links so the published
reference does not contain dead anchors.
"""

import re
import sys

V1BETA1_MARKER = "## nvidia.com/v1beta1"
TYPE_HEADING_PATTERN = re.compile(r"^####\s+(.+?)\s*$", re.MULTILINE)


def type_headings(content: str) -> set[str]:
    """Return the generated type names from fourth-level headings."""
    return set(TYPE_HEADING_PATTERN.findall(content))


def fix_api_anchors(content: str) -> str:
    """Namespace v1beta1 anchors for types also present in v1alpha1."""
    idx = content.find(V1BETA1_MARKER)
    if idx == -1:
        return content

    alpha_part = content[:idx]
    beta_part = content[idx:]

    for type_name in type_headings(alpha_part) & type_headings(beta_part):
        anchor = type_name.lower()
        beta_part = re.sub(
            r"(####\s+)" + re.escape(type_name) + r"(\s*$)",
            r"\1v1beta1 " + type_name + r"\2",
            beta_part,
            flags=re.MULTILINE,
        )
        beta_part = beta_part.replace(f"(#{anchor})", f"(#v1beta1-{anchor})")

    return alpha_part + beta_part


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <api-reference.md>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as file:
        content = file.read()

    if V1BETA1_MARKER not in content:
        print(
            "Warning: v1beta1 section not found, skipping anchor fix", file=sys.stderr
        )
        return

    content = fix_api_anchors(content)

    external_types_without_sections = [
        "EndpointPickerConfig",
    ]
    for type_name in external_types_without_sections:
        anchor = type_name.lower()
        content = content.replace(f"[{type_name}](#{anchor})", type_name)

    with open(path, "w") as file:
        file.write(content)
    print(f"✅ Fixed duplicate anchors in {path}")


if __name__ == "__main__":
    main()

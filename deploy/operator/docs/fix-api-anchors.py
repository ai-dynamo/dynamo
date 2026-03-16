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

"""Post-process api-reference.md to deduplicate anchors between v1alpha1 and v1beta1.

crd-ref-docs generates anchors solely from type names, so types that exist in both
API versions get identical anchors (e.g. #dynamographdeploymentrequest). In standard
Markdown renderers the first occurrence wins, meaning v1beta1 links resolve to the
v1alpha1 section. This script prepends "v1beta1 " to the affected headings in the
v1beta1 section and updates all intra-section links to match the new anchors.
"""
import re
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <api-reference.md>", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
content = open(path).read()

marker = "## nvidia.com/v1beta1"
idx = content.find(marker)
if idx == -1:
    print("Warning: v1beta1 section not found, skipping anchor fix", file=sys.stderr)
    sys.exit(0)

alpha_part = content[:idx]
beta_part = content[idx:]

# Types whose names collide between v1alpha1 and v1beta1.
# Add to this list if future versions introduce additional same-named types.
duplicate_types = [
    "DynamoGraphDeploymentRequest",
    "DynamoGraphDeploymentRequestSpec",
    "DynamoGraphDeploymentRequestStatus",
]

for t in duplicate_types:
    anchor = t.lower()
    # Rename section headings: #### TypeName → #### v1beta1 TypeName
    beta_part = re.sub(
        r"(####\s+)" + re.escape(t) + r"(\s*$)",
        r"\1v1beta1 " + t + r"\2",
        beta_part,
        flags=re.MULTILINE,
    )
    # Update markdown links: (#anchor) → (#v1beta1-anchor)
    beta_part = beta_part.replace(f"(#{anchor})", f"(#v1beta1-{anchor})")

open(path, "w").write(alpha_part + beta_part)
print(f"✅ Fixed duplicate anchors in {path}")

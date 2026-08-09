# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest


FIX_API_ANCHORS = runpy.run_path(
    Path(__file__).parents[2] / "deploy/operator/docs/fix-api-anchors.py"
)["fix_api_anchors"]


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_fix_api_anchors_namespaces_all_colliding_v1beta1_types() -> None:
    content = """## nvidia.com/v1alpha1

#### DynamoGraphDeployment

| `spec` _[DynamoGraphDeploymentSpec](#dynamographdeploymentspec)_ |

#### DynamoGraphDeploymentSpec

#### V1AlphaOnly

## nvidia.com/v1beta1

### Resource Types
- [DynamoGraphDeployment](#dynamographdeployment)
- [DynamoGraphDeploymentSpec](#dynamographdeploymentspec)
- [V1BetaOnly](#v1betaonly)

#### DynamoGraphDeployment

| `spec` _[DynamoGraphDeploymentSpec](#dynamographdeploymentspec)_ |

#### DynamoGraphDeploymentSpec

#### V1BetaOnly
"""

    fixed = FIX_API_ANCHORS(content)

    alpha_part, beta_part = fixed.split("## nvidia.com/v1beta1", maxsplit=1)
    assert "#### DynamoGraphDeployment" in alpha_part
    assert "(#dynamographdeploymentspec)" in alpha_part
    assert "#### v1beta1 DynamoGraphDeployment" in beta_part
    assert "#### v1beta1 DynamoGraphDeploymentSpec" in beta_part
    assert "(#v1beta1-dynamographdeployment)" in beta_part
    assert "(#v1beta1-dynamographdeploymentspec)" in beta_part
    assert "#### V1BetaOnly" in beta_part
    assert "(#v1betaonly)" in beta_part

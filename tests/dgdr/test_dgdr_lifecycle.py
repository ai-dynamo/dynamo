# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DGDR v1beta1 lifecycle tests.

Scope for PR2:
- Keep lifecycle coverage small and stable.
- Focus deep profiling behavior in test_dgdr_profiling.py.
"""

from __future__ import annotations

import pytest

from tests.dgdr.conftest import (
    DEFAULT_DEPLOY_TIMEOUT,
    DEFAULT_PROFILING_TIMEOUT,
    PHASE_DEPLOYED,
    PHASE_READY,
    build_dgdr_manifest,
    unique_dgdr_name,
)
from tests.utils.managed_deployment import ManagedDGDR

READY_TEST_TIMEOUT_S = 3 * DEFAULT_PROFILING_TIMEOUT
DEPLOY_TEST_TIMEOUT_S = 3 * (DEFAULT_PROFILING_TIMEOUT + DEFAULT_DEPLOY_TIMEOUT)


@pytest.mark.gpu_0
@pytest.mark.nightly
@pytest.mark.integration
@pytest.mark.k8s
class TestDGDRLifecycle:
    @pytest.mark.timeout(READY_TEST_TIMEOUT_S)
    def test_rapid_autoapply_false_reaches_ready(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_factory,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_profiling_timeout: int,
    ) -> None:
        """Rapid profiling with autoApply=false should complete to Ready."""
        name = unique_dgdr_name("lifecycle-ready")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            auto_apply=False,
        )
        dgdr_factory(manifest)

        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name, PHASE_READY, timeout=dgdr_profiling_timeout
            )
        )
        obj = managed_dgdr.run(managed_dgdr.get(name))
        assert obj is not None
        assert obj.get("status", {}).get("phase") == PHASE_READY
        assert obj.get("status", {}).get("profilingJobName")

    @pytest.mark.timeout(DEPLOY_TEST_TIMEOUT_S)
    def test_rapid_autoapply_true_reaches_deployed_without_mocker(
        self,
        managed_dgdr: ManagedDGDR,
        dgdr_factory,
        dgdr_image: str,
        dgdr_model: str,
        dgdr_profiling_timeout: int,
        dgdr_deploy_timeout: int,
        dgdr_use_mocker: bool,
    ) -> None:
        """On real-GPU mode, autoApply=true should progress to Deployed."""
        if dgdr_use_mocker:
            pytest.skip(
                "In mocker mode, autoApply=true can race on generated DGD; "
                "lifecycle deployment coverage is run with --dgdr-no-mocker"
            )

        name = unique_dgdr_name("lifecycle-deployed")
        manifest = build_dgdr_manifest(
            name,
            model=dgdr_model,
            image=dgdr_image,
            backend="vllm",
            search_strategy="rapid",
            auto_apply=True,
        )
        dgdr_factory(manifest)

        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name,
                PHASE_DEPLOYED,
                timeout=dgdr_profiling_timeout + dgdr_deploy_timeout,
            )
        )
        obj = managed_dgdr.run(managed_dgdr.get(name))
        assert obj is not None
        status = obj.get("status", {})
        assert status.get("phase") == PHASE_DEPLOYED
        assert status.get("dgdName")

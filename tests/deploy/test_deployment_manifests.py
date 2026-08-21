# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks that the deploy suites point at manifests that still exist.

The deploy tests only run against a live cluster, so a manifest that moves is
not noticed until a cluster job fails minutes in -- which is how
``examples/backends/*/deploy/v1beta1/agg.yaml`` stayed broken after the
manifests were promoted a directory up. These tests need no cluster and no GPU,
so the same breakage now shows up in pre-merge instead.
"""

import pytest

from tests.deploy.test_dynamocheckpoint import (
    CHECKPOINT_BACKENDS,
    _checkpoint_manifest_path,
    _new_checkpoint_spec,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]

BACKENDS = sorted(CHECKPOINT_BACKENDS)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_checkpoint_manifest_exists(backend_name):
    path = _checkpoint_manifest_path(CHECKPOINT_BACKENDS[backend_name])
    assert path.is_file(), (
        f"DynamoCheckpoint manifest for {backend_name} not found: {path}. "
        "If the example manifests moved, update CHECKPOINT_BACKENDS[...].manifest."
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_checkpoint_spec_builds_and_declares_the_expected_components(backend_name):
    """The spec must carry the components the test later scales and forwards to."""
    backend = CHECKPOINT_BACKENDS[backend_name]

    spec = _new_checkpoint_spec(
        backend=backend,
        name=f"{backend_name}-checkpoint-spec-test",
        namespace="default",
        image="registry.invalid/dynamo:test",
        frontend_image="registry.invalid/dynamo:frontend",
        model_cache_pvc=None,
        model_cache_mount=None,
    )

    components = [service.name for service in spec.services]
    assert backend.decode_component in components, (
        f"{backend_name}: decode component {backend.decode_component!r} missing "
        f"from manifest components {components}"
    )
    assert backend.frontend_component in components, (
        f"{backend_name}: frontend component {backend.frontend_component!r} "
        f"missing from manifest components {components}"
    )
    # frontend_endpoint() forwards this port and appends this path.
    assert spec.port > 0
    assert spec.endpoint.startswith("/")

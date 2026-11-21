# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Options and cluster fixtures for the multi-cluster Global Discovery test.

The hub cluster runs the Global Discovery Service; the member cluster runs the
pool. ``--namespace`` (default ``default``) is the namespace used in both.
"""

import os
from typing import Optional

import pytest

from tests.deploy.global_discovery.cluster import Cluster


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup(
        "global-discovery", "multi-cluster Global Discovery deploy test"
    )
    group.addoption(
        "--hub-kubeconfig",
        default=os.environ.get("KUBECONFIG"),
        help="Kubeconfig of the hub cluster, which runs the Global Discovery "
        "Service. Default: $KUBECONFIG, then kubectl's own default.",
    )
    group.addoption(
        "--member-kubeconfig",
        default=os.environ.get("GLOBAL_DISCOVERY_MEMBER_KUBECONFIG"),
        help="Kubeconfig of the member cluster, which runs the pool and the "
        "Dynamo operator. Default: $GLOBAL_DISCOVERY_MEMBER_KUBECONFIG.",
    )
    group.addoption(
        "--global-discovery-image",
        default=os.environ.get("GLOBAL_DISCOVERY_IMAGE"),
        help="Global Discovery Service image, built from "
        "container/Dockerfile.global-discovery-service. "
        "Default: $GLOBAL_DISCOVERY_IMAGE.",
    )


def _required(request: pytest.FixtureRequest, option: str) -> str:
    value: Optional[str] = request.config.getoption(option)
    if not value:
        pytest.fail(f"{option} is required for this test", pytrace=False)
    return value


@pytest.fixture
def test_namespace(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--namespace") or "default"


@pytest.fixture
def hub(request: pytest.FixtureRequest, test_namespace: str) -> Cluster:
    return Cluster(
        role="hub",
        kubeconfig=request.config.getoption("--hub-kubeconfig"),
        namespace=test_namespace,
    )


@pytest.fixture
def member(request: pytest.FixtureRequest, test_namespace: str) -> Cluster:
    return Cluster(
        role="member",
        kubeconfig=_required(request, "--member-kubeconfig"),
        namespace=test_namespace,
    )


@pytest.fixture
def global_discovery_image(request: pytest.FixtureRequest) -> str:
    return _required(request, "--global-discovery-image")


@pytest.fixture
def pool_image(request: pytest.FixtureRequest) -> str:
    """Image for the pool's frontend, mocker worker and PoolRelay.

    The frontend image carries the whole ai-dynamo wheel, so it runs
    ``dynamo.frontend``, ``dynamo.mocker`` and ``dynamo.pool_relay``.
    """
    return _required(request, "--frontend-image")

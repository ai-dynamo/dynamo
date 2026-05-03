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

import os

import pytest

from tests.fault_tolerance.deploy.scenarios import scenarios


@pytest.fixture(autouse=True)
def _route_outputs_to_repo(monkeypatch):
    """Route this dir's tests' outputs to ``<cwd>/test_outputs/<test>/``.

    The shared ``resolve_test_output_path`` defaults to
    ``/tmp/dynamo_tests/`` to keep outputs out of the git tree, which is
    fine when pytest runs on the host. Inside a dev container with the
    worktree mounted at ``/workspace``, ``/tmp`` is container-local and
    artifacts disappear from the host. We narrow the override to this
    conftest so non-k8s tests on the host keep their existing default,
    and k8s tests get one host-visible per-test directory holding load,
    services, pod yaml, test.log.txt, and the report together.

    The user-set ``DYN_TEST_OUTPUT_PATH`` always wins (we use
    ``setdefault`` semantics via the ``not in`` guard).
    """
    if "DYN_TEST_OUTPUT_PATH" not in os.environ:
        monkeypatch.setenv(
            "DYN_TEST_OUTPUT_PATH",
            os.path.join(os.getcwd(), "test_outputs"),
        )
    yield


# Shared CLI options (--image, --namespace, --skip-service-restart) are defined in tests/conftest.py.
# Only fault_tolerance-specific options are defined here.
def pytest_addoption(parser):
    parser.addoption(
        "--client-type",
        type=str,
        default=None,
        choices=["aiperf", "legacy"],
        help="Client type for load generation: 'aiperf' (default) or 'legacy'",
    )
    parser.addoption(
        "--include-custom-build",
        action="store_true",
        default=False,
        help="Include tests that require custom builds (e.g., MoE models). "
        "By default, these tests are excluded.",
    )
    parser.addoption(
        "--storage-class",
        type=str,
        default=None,
        help="Storage class for PVC log collection (must support RWX). "
        "If not specified, uses cluster default.",
    )
    parser.addoption(
        "--restart-services",
        action="store_true",
        default=False,
        help="Restart NATS and etcd before each test. "
        "Default is to skip restart (faster iteration).",
    )


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests and apply markers based on scenario properties.

    This hook applies markers to individual test instances based on their scenario:
    - @pytest.mark.custom_build: For MoE models and other tests requiring custom builds
    """
    if "scenario" in metafunc.fixturenames:
        scenario_names = list(scenarios.keys())
        argvalues = []
        ids = []

        for scenario_name in scenario_names:
            scenario_obj = scenarios[scenario_name]
            marks = []

            if getattr(scenario_obj, "requires_custom_build", False):
                marks.append(pytest.mark.custom_build)

            # Always use pytest.param for type consistency (even with empty marks)
            argvalues.append(pytest.param(scenario_name, marks=marks))
            ids.append(scenario_name)

        metafunc.parametrize("scenario_name", argvalues, ids=ids)


def pytest_collection_modifyitems(config, items):
    """Automatically deselect custom_build tests unless --include-custom-build is specified.

    This allows users to run tests without any special flags and automatically excludes
    tests that require custom builds. To include them, use --include-custom-build.

    Note: If user explicitly uses -m marker filtering, we respect that and don't
    auto-deselect, allowing them to run custom_build tests with -m "custom_build".
    """
    # If --include-custom-build flag is set, include all tests
    if config.getoption("--include-custom-build"):
        return

    # If user explicitly used -m marker filtering, let pytest handle it
    # Don't auto-deselect in this case
    if config.option.markexpr:
        return

    # Default case: auto-deselect custom_build tests
    deselected = []
    selected = []

    for item in items:
        if "custom_build" in item.keywords:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def image(request):
    return request.config.getoption("--image")


@pytest.fixture
def namespace(request):
    """Get Kubernetes namespace from CLI option, with fault-tolerance-specific default."""
    value = request.config.getoption("--namespace")
    return value if value is not None else "fault-tolerance-test"


@pytest.fixture
def client_type(request):
    """Get client type from command line or use scenario default."""
    return request.config.getoption("--client-type")


@pytest.fixture
def skip_service_restart(request):
    """Whether to skip restarting NATS and etcd services.

    Default: True (skip restart — services are assumed to be running).
    Pass --restart-services to opt in to a clean-state restart.
    The legacy --skip-service-restart flag is still honored if explicitly set.
    """
    if request.config.getoption("--restart-services", default=False):
        return False  # don't skip = do restart
    skip = request.config.getoption("--skip-service-restart", default=None)
    if skip is not None:
        return skip
    return True  # default: skip restart


@pytest.fixture
def storage_class(request):
    """Storage class for PVC log collection (must support RWX)."""
    return request.config.getoption("--storage-class")

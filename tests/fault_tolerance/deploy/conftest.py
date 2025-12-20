# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from tests.fault_tolerance.deploy.scenarios import scenarios


def pytest_addoption(parser):
    parser.addoption("--image", type=str, default=None)
    parser.addoption("--namespace", type=str, default="fault-tolerance-test")
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
    return request.config.getoption("--namespace")


@pytest.fixture
def client_type(request):
    """Get client type from command line or use scenario default."""
    return request.config.getoption("--client-type")

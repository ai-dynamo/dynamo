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

"""
Pytest configuration for deployment tests.

This module provides dynamic test discovery and fixtures for running deployment tests
against Kubernetes deployments. It supports discovering deployments from different
sources (e.g., examples, recipes) through a pluggable discovery system.

Architecture Overview
---------------------

The test infrastructure is built around three key concepts:

1. **DeploymentTarget**: A simple data class representing a single deployment
   to test. Contains all information needed to locate and identify the deployment.

2. **Discovery Functions**: Functions that scan the filesystem and return a list
   of DeploymentTarget objects. Each source type (examples, recipes) has its own
   discovery function.

3. **Test Parametrization**: Tests are parametrized with DeploymentTarget objects,
   making them source-agnostic. The test only cares about having a valid YAML path.

Extension Points for Future Recipe Support
------------------------------------------

To add recipe support, you would:

1. Create a new discovery function (e.g., `discover_recipe_targets()`) that scans
   the `recipes/` directory and returns DeploymentTarget objects.

2. Add new CLI options if needed (e.g., `--model`, `--source-type`).

3. Update `_collect_all_targets()` to include recipe targets.

4. Optionally add filtering logic in `pytest_generate_tests()` for recipe-specific
   filtering (by model, mode, etc.).

The test itself (`test_deploy.py`) should require no changes because it works
with the DeploymentTarget abstraction, not specific source paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from tests.utils.managed_deployment import DeploymentSpec


def _get_workspace_dir() -> Path:
    """Get the workspace root directory containing pyproject.toml."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume workspace is 3 levels up from tests/deploy/
    return Path(__file__).resolve().parent.parent.parent


# -----------------------------------------------------------------------------
# DeploymentTarget: The core abstraction for test parametrization
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentTarget:
    """Represents a deployment configuration to be tested.

    This is the central abstraction that makes the test infrastructure
    source-agnostic. Whether a deployment comes from examples/ or recipes/,
    the test sees the same interface.

    Attributes:
        yaml_path: Absolute path to the deployment YAML file
        framework: The inference framework (vllm, sglang, trtllm, etc.)
        profile: The deployment profile name (agg, disagg, etc.)
        source: Where this target came from (examples, recipes)

    Future attributes for recipe support:
        model: The model name (e.g., "llama-3-70b", "deepseek-r1")
        mode: The deployment mode (e.g., "agg", "disagg-single-node")
    """

    yaml_path: Path
    framework: str
    profile: str
    source: str = "examples"

    @property
    def test_id(self) -> str:
        """Generate a unique, readable test ID for pytest parametrization."""
        return f"{self.framework}-{self.profile}"

    def exists(self) -> bool:
        """Check if the deployment YAML file exists."""
        return self.yaml_path.exists()


# -----------------------------------------------------------------------------
# Discovery Functions: Finding deployment targets from various sources
# -----------------------------------------------------------------------------


def discover_example_targets(workspace: Optional[Path] = None) -> List[DeploymentTarget]:
    """Discover deployment targets from examples/backends/{framework}/deploy/*.yaml.

    This function scans the examples directory for deployment YAML files.
    Files in subdirectories (e.g., lora/) are excluded.

    Args:
        workspace: Workspace root directory. If None, auto-detected.

    Returns:
        List of DeploymentTarget objects for each discovered deployment.
    """
    if workspace is None:
        workspace = _get_workspace_dir()

    backends_dir = workspace / "examples" / "backends"
    targets: List[DeploymentTarget] = []

    if not backends_dir.exists():
        return targets

    for framework_dir in backends_dir.iterdir():
        if not framework_dir.is_dir():
            continue

        deploy_dir = framework_dir / "deploy"
        if not deploy_dir.exists():
            continue

        framework_name = framework_dir.name

        for yaml_file in deploy_dir.glob("*.yaml"):
            # Only include files directly in deploy/, not in subdirectories
            if yaml_file.parent != deploy_dir:
                continue

            profile_name = yaml_file.stem
            targets.append(
                DeploymentTarget(
                    yaml_path=yaml_file,
                    framework=framework_name,
                    profile=profile_name,
                    source="examples",
                )
            )

    return targets


# Future: Add recipe discovery function here
#
# def discover_recipe_targets(
#     workspace: Optional[Path] = None,
#     model_filter: Optional[str] = None,
# ) -> List[DeploymentTarget]:
#     """Discover deployment targets from recipes/{model}/{framework}/{mode}/deploy.yaml.
#
#     Args:
#         workspace: Workspace root directory. If None, auto-detected.
#         model_filter: Optional filter to only include specific model recipes.
#
#     Returns:
#         List of DeploymentTarget objects for each discovered recipe deployment.
#     """
#     if workspace is None:
#         workspace = _get_workspace_dir()
#
#     recipes_dir = workspace / "recipes"
#     targets: List[DeploymentTarget] = []
#
#     if not recipes_dir.exists():
#         return targets
#
#     # Scan for deploy*.yaml files in recipe directories
#     for deploy_yaml in recipes_dir.glob("**/deploy*.yaml"):
#         # Parse path: recipes/{model}/{framework}/{mode}/deploy.yaml
#         relative = deploy_yaml.relative_to(recipes_dir)
#         parts = relative.parts
#
#         if len(parts) < 4:
#             continue
#
#         model_name, framework, mode = parts[0], parts[1], parts[2]
#
#         if model_filter and model_name != model_filter:
#             continue
#
#         # Construct a descriptive profile name from the path
#         profile = f"{model_name}/{mode}"
#         if deploy_yaml.stem != "deploy":
#             # Handle variants like deploy_hopper_16gpu.yaml
#             profile = f"{profile}-{deploy_yaml.stem.replace('deploy_', '')}"
#
#         targets.append(
#             DeploymentTarget(
#                 yaml_path=deploy_yaml,
#                 framework=framework,
#                 profile=profile,
#                 source="recipes",
#             )
#         )
#
#     return targets


def _collect_all_targets() -> List[DeploymentTarget]:
    """Collect deployment targets from all sources.

    This is the single point where all discovery functions are aggregated.
    To add a new source, add its discovery function call here.

    Returns:
        Combined list of all deployment targets, sorted by test_id.
    """
    targets: List[DeploymentTarget] = []

    # Discover from examples
    targets.extend(discover_example_targets())

    # Future: Uncomment to enable recipe discovery
    # targets.extend(discover_recipe_targets())

    # Sort for consistent test ordering
    return sorted(targets, key=lambda t: (t.source, t.framework, t.profile))


def _build_test_matrix(targets: List[DeploymentTarget]) -> Dict[str, List[str]]:
    """Build a framework -> profiles mapping for CLI validation.

    This preserves backward compatibility with the existing CLI interface
    that validates --framework and --profile options.

    Args:
        targets: List of deployment targets to index

    Returns:
        Dictionary mapping framework names to lists of profile names.
    """
    matrix: Dict[str, List[str]] = {}
    for target in targets:
        if target.framework not in matrix:
            matrix[target.framework] = []
        if target.profile not in matrix[target.framework]:
            matrix[target.framework].append(target.profile)

    # Sort profiles within each framework
    for framework in matrix:
        matrix[framework] = sorted(matrix[framework])

    return matrix


# Discover all targets and build matrix at module load time for test collection
ALL_DEPLOYMENT_TARGETS = _collect_all_targets()
DEPLOY_TEST_MATRIX = _build_test_matrix(ALL_DEPLOYMENT_TARGETS)


# -----------------------------------------------------------------------------
# Pytest Configuration: CLI options and test parametrization
# -----------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for deployment tests."""
    parser.addoption(
        "--image",
        type=str,
        default=None,
        help="Container image to use for deployment (overrides YAML default)",
    )
    parser.addoption(
        "--namespace",
        type=str,
        default="deploy-test",
        help="Kubernetes namespace for deployment",
    )
    parser.addoption(
        "--framework",
        type=str,
        default=None,
        choices=list(DEPLOY_TEST_MATRIX.keys()) if DEPLOY_TEST_MATRIX else None,
        help="Framework to test (e.g., vllm, sglang, trtllm)",
    )
    parser.addoption(
        "--profile",
        type=str,
        default=None,
        help="Deployment profile to test (e.g., agg, disagg, disagg_router)",
    )
    parser.addoption(
        "--skip-service-restart",
        action="store_true",
        default=True,
        help="Skip restarting NATS and etcd services before deployment (default: True). "
        "Use --no-skip-service-restart to restart services.",
    )
    # Future: Add recipe-specific options here
    # parser.addoption(
    #     "--model",
    #     type=str,
    #     default=None,
    #     help="Model name to test (for recipe deployments)",
    # )
    # parser.addoption(
    #     "--source",
    #     type=str,
    #     default=None,
    #     choices=["examples", "recipes"],
    #     help="Source type to filter (examples or recipes)",
    # )


def _filter_targets(
    targets: List[DeploymentTarget],
    framework: Optional[str] = None,
    profile: Optional[str] = None,
) -> List[DeploymentTarget]:
    """Filter deployment targets based on CLI options.

    This centralizes filtering logic, making it easy to add new filters
    for recipe support (e.g., model, source, mode).

    Args:
        targets: List of targets to filter
        framework: Optional framework filter
        profile: Optional profile filter

    Returns:
        Filtered list of targets
    """
    result = targets

    if framework:
        result = [t for t in result if t.framework == framework]

    if profile:
        result = [t for t in result if t.profile == profile]

    # Future: Add more filters here
    # if model:
    #     result = [t for t in result if getattr(t, 'model', None) == model]
    # if source:
    #     result = [t for t in result if t.source == source]

    return result


def _find_target(
    framework: str, profile: str, targets: List[DeploymentTarget]
) -> Optional[DeploymentTarget]:
    """Find a specific deployment target by framework and profile.

    Args:
        framework: Framework name to match
        profile: Profile name to match
        targets: List of targets to search

    Returns:
        Matching DeploymentTarget or None if not found
    """
    for target in targets:
        if target.framework == framework and target.profile == profile:
            return target
    return None


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on CLI options or full matrix.

    If --framework and --profile are specified, runs only that combination.
    Otherwise, generates tests for the full matrix of discovered deployments.

    The test receives both the DeploymentTarget and individual parameters
    (framework, profile) for backward compatibility and readable test output.
    """
    if "deployment_target" not in metafunc.fixturenames:
        return

    framework_opt = metafunc.config.getoption("--framework")
    profile_opt = metafunc.config.getoption("--profile")

    # Filter targets based on CLI options
    filtered_targets = _filter_targets(
        ALL_DEPLOYMENT_TARGETS,
        framework=framework_opt,
        profile=profile_opt,
    )

    # Validate that requested combination exists
    if framework_opt and profile_opt and not filtered_targets:
        if framework_opt not in DEPLOY_TEST_MATRIX:
            pytest.skip(f"Framework '{framework_opt}' not found in discovered profiles")
            return
        if profile_opt not in DEPLOY_TEST_MATRIX.get(framework_opt, []):
            pytest.skip(
                f"Profile '{profile_opt}' not found for framework '{framework_opt}'"
            )
            return

    # Build parametrization
    if filtered_targets:
        metafunc.parametrize(
            "deployment_target",
            filtered_targets,
            ids=[t.test_id for t in filtered_targets],
        )


# -----------------------------------------------------------------------------
# Fixtures: Providing test dependencies
# -----------------------------------------------------------------------------


@pytest.fixture
def image(request: pytest.FixtureRequest) -> Optional[str]:
    """Get custom container image from CLI option."""
    return request.config.getoption("--image")


@pytest.fixture
def namespace(request: pytest.FixtureRequest) -> str:
    """Get Kubernetes namespace from CLI option."""
    return request.config.getoption("--namespace")


@pytest.fixture
def skip_service_restart(request: pytest.FixtureRequest) -> bool:
    """Get skip_service_restart flag from CLI option."""
    return request.config.getoption("--skip-service-restart")


@pytest.fixture
def framework(deployment_target: DeploymentTarget) -> str:
    """Extract framework from deployment target for backward compatibility."""
    return deployment_target.framework


@pytest.fixture
def profile(deployment_target: DeploymentTarget) -> str:
    """Extract profile from deployment target for backward compatibility."""
    return deployment_target.profile


@pytest.fixture
def deployment_yaml(deployment_target: DeploymentTarget) -> Path:
    """Get the path to deployment YAML file from the target.

    This fixture validates that the YAML file exists before returning.
    """
    yaml_path = deployment_target.yaml_path

    if not yaml_path.exists():
        pytest.fail(f"Deployment YAML not found: {yaml_path}")

    return yaml_path


@pytest.fixture
def deployment_spec(
    deployment_yaml: Path,
    image: Optional[str],
    namespace: str,
) -> DeploymentSpec:
    """Create DeploymentSpec from YAML with optional image override.

    This fixture is source-agnostic - it works the same way whether
    the YAML comes from examples or recipes.

    Args:
        deployment_yaml: Path to the deployment YAML file
        image: Optional container image override
        namespace: Kubernetes namespace for deployment

    Returns:
        Configured DeploymentSpec ready for deployment
    """
    spec = DeploymentSpec(str(deployment_yaml))

    # Set namespace
    spec.namespace = namespace

    # Override image if provided
    if image:
        spec.set_image(image)

    return spec

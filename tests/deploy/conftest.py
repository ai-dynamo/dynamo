# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for deployment tests.

This module provides dynamic test discovery and fixtures for running deployment tests
against Kubernetes deployments. This currently only covers deployments in the examples directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from tests.utils.managed_deployment import DeploymentSpec, _get_workspace_dir


# Shared CLI options (--image, --namespace, --skip-service-restart) are defined in tests/conftest.py.
# Only deploy-specific options are defined here.
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add deploy-specific command-line options.

    These options control which deployment configurations are tested.
    Shared options (--image, --namespace, --skip-service-restart) are
    defined in tests/conftest.py.
    """
    parser.addoption(
        "--framework",
        type=str,
        default=None,
        help="Framework to test (e.g., vllm, sglang, trtllm). "
        "If not specified, runs all discovered frameworks.",
    )
    parser.addoption(
        "--profile",
        type=str,
        default=None,
        help="Deployment profile to test (e.g., agg, disagg, disagg_router). "
        "If not specified, runs all profiles for the selected framework.",
    )


@dataclass(frozen=True)
class DeploymentTarget:
    """Represents a deployment configuration to be tested.

    Attributes:
        yaml_path: Absolute path to the deployment YAML file
        framework: The inference framework (vllm, sglang, trtllm, etc.)
        profile: The deployment profile name (agg, disagg, etc.)
        source: Where this target came from (e.g., examples)
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


def discover_example_targets(
    workspace: Optional[Path] = None,
) -> List[DeploymentTarget]:
    """Discover deployment targets from examples/backends/{framework}/deploy/*.yaml.

    This function scans the examples directory for deployment YAML files.
    Files in subdirectories (e.g., lora/) are excluded.

    Args:
        workspace: Workspace root directory. If None, auto-detected.

    Returns:
        List of DeploymentTarget objects for each discovered deployment.
    """
    if workspace is None:
        workspace = Path(_get_workspace_dir())

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


def _collect_all_targets() -> List[DeploymentTarget]:
    """Collect deployment targets from all sources.

    Returns:
        List of all deployment targets, sorted for consistent test ordering.
    """
    targets: List[DeploymentTarget] = []

    # Discover from examples
    targets.extend(discover_example_targets())

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


def _filter_targets(
    targets: List[DeploymentTarget],
    framework: Optional[str] = None,
    profile: Optional[str] = None,
) -> List[DeploymentTarget]:
    """Filter deployment targets based on CLI options.

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
    """Whether to skip restarting NATS and etcd services.

    Deploy tests default to SKIPPING restart (for speed).
    The --skip-service-restart flag can override this behavior.

    Returns:
        If --skip-service-restart is passed: True (skip restart)
        If flag not passed: True (deploy tests skip by default)
    """
    value = request.config.getoption("--skip-service-restart")
    return value if value is not None else True  # Default: skip for deploy tests


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

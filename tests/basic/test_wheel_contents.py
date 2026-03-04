# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, files

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]


def test_no_bundled_shared_libraries():
    """Ensure ai-dynamo-runtime does not bundle any shared libraries.

    All .so dependencies should come from system installs or pip packages,
    not be bundled inside the wheel by auditwheel.  If this test fails,
    add --exclude flags to the auditwheel repair command in
    container/templates/wheel_builder.Dockerfile.
    """
    try:
        installed_files = files("ai-dynamo-runtime")
    except PackageNotFoundError:
        pytest.fail("ai-dynamo-runtime is not installed")

    bundled_libs = [
        str(f) for f in installed_files if ".libs/" in str(f) and ".so" in str(f)
    ]

    assert (
        not bundled_libs
    ), "Unexpected shared libraries bundled in ai-dynamo-runtime:\n" + "\n".join(
        f"  {lib}" for lib in bundled_libs
    )

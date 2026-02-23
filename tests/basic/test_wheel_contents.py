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

"""Test that ai-dynamo-runtime does not bundle shared libraries."""

from importlib.metadata import files

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
    installed_files = files("ai-dynamo-runtime")
    assert installed_files is not None, "ai-dynamo-runtime is not installed"

    bundled_libs = [
        str(f) for f in installed_files if ".libs/" in str(f) and ".so" in str(f)
    ]

    assert not bundled_libs, (
        "Unexpected shared libraries bundled in ai-dynamo-runtime:\n"
        + "\n".join(f"  {lib}" for lib in bundled_libs)
        + "\nAdd --exclude flags to auditwheel repair in "
        "container/templates/wheel_builder.Dockerfile"
    )

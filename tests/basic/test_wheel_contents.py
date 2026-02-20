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

"""Test that the ai-dynamo-runtime wheel does not bundle nixl shared libraries."""

import importlib.metadata

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]


NIXL_LIBS = {"libnixl.so", "libnixl_build.so", "libnixl_common.so", "libstream.so", "libserdes.so"}


def test_no_bundled_nixl_libraries():
    """Ensure ai-dynamo-runtime wheel does not bundle nixl shared libraries.

    Nixl .so files should come from pip install nixl, not be bundled inside
    the wheel by auditwheel.  If this test fails, check the --exclude flags
    in the auditwheel repair command in
    container/templates/wheel_builder.Dockerfile.
    """
    dist = importlib.metadata.distribution("ai-dynamo-runtime")
    files = dist.files
    if files is None:
        pytest.skip("No RECORD available for ai-dynamo-runtime (editable install?)")

    bundled_nixl = [
        str(f)
        for f in files
        if "ai_dynamo_runtime.libs" in str(f)
        and any(name in str(f) for name in NIXL_LIBS)
    ]
    assert not bundled_nixl, (
        "Nixl shared libraries should not be bundled in ai-dynamo-runtime wheel:\n"
        + "\n".join(f"  {lib}" for lib in bundled_nixl)
        + "\nCheck --exclude flags in auditwheel repair in "
        "container/templates/wheel_builder.Dockerfile"
    )

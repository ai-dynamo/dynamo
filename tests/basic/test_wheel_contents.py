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

"""Test that the ai-dynamo-runtime wheel does not bundle shared libraries."""

import glob
import zipfile

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]

WHEEL_DIRS = ["/opt/dynamo/wheelhouse", "/opt/dynamo/dist"]


def _find_runtime_wheel():
    """Find the ai-dynamo-runtime wheel file on disk."""
    for d in WHEEL_DIRS:
        matches = glob.glob(f"{d}/ai_dynamo_runtime-*.whl")
        if matches:
            return matches[0]
    return None


def test_no_bundled_shared_libraries():
    """Ensure ai-dynamo-runtime wheel does not bundle any shared libraries.

    All .so dependencies should come from system installs or pip packages,
    not be bundled inside the wheel by auditwheel.  If this test fails,
    add --exclude flags to the auditwheel repair command in
    container/templates/wheel_builder.Dockerfile.
    """
    whl_path = _find_runtime_wheel()
    if whl_path is None:
        pytest.skip("ai-dynamo-runtime wheel not found in " + ", ".join(WHEEL_DIRS))

    with zipfile.ZipFile(whl_path) as zf:
        bundled_libs = [
            name for name in zf.namelist() if ".libs/" in name and ".so" in name
        ]

    assert not bundled_libs, (
        f"Unexpected shared libraries bundled in {whl_path}:\n"
        + "\n".join(f"  {lib}" for lib in bundled_libs)
        + "\nAdd --exclude flags to auditwheel repair in "
        "container/templates/wheel_builder.Dockerfile"
    )

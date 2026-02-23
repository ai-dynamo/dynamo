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

WHEEL_DIR = "/opt/dynamo/wheelhouse"


def test_no_bundled_shared_libraries():
    """Ensure ai-dynamo-runtime wheel does not bundle any shared libraries.

    All .so dependencies should come from system installs or pip packages,
    not be bundled inside the wheel by auditwheel.  If this test fails,
    add --exclude flags to the auditwheel repair command in
    container/templates/wheel_builder.Dockerfile.
    """
    matches = glob.glob(f"{WHEEL_DIR}/ai_dynamo_runtime-*.whl")
    assert matches, (
        f"ai-dynamo-runtime wheel not found in {WHEEL_DIR}"
    )
    whl_path = matches[0]

    with zipfile.ZipFile(whl_path) as zf:
        bundled_libs = [
            name for name in zf.namelist()
            if ".libs/" in name and ".so" in name
        ]

    assert not bundled_libs, (
        f"Unexpected shared libraries bundled in {whl_path}:\n"
        + "\n".join(f"  {lib}" for lib in bundled_libs)
        + "\nAdd --exclude flags to auditwheel repair in "
        "container/templates/wheel_builder.Dockerfile"
    )

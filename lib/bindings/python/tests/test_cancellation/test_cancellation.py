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

import os
import subprocess

import pytest

pytestmark = pytest.mark.pre_merge


def _run_test_in_subprocess(test_name: str):
    """Helper function to run a test file in a separate process"""
    test_file = os.path.join(os.path.dirname(__file__), f"{test_name}.py")
    result = subprocess.run(
        ["pytest", test_file, "-v"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    assert (
        result.returncode == 0
    ), f"Test {test_name} failed with return code {result.returncode}"


def test_client_context_cancel():
    _run_test_in_subprocess("test_client_context_cancel")


def test_client_loop_break():
    _run_test_in_subprocess("test_client_loop_break")


def test_server_context_cancel():
    _run_test_in_subprocess("test_server_context_cancel")


def test_server_raise_cancelled():
    _run_test_in_subprocess("test_server_raise_cancelled")

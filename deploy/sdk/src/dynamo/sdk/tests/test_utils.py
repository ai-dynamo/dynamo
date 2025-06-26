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

from dynamo.sdk.lib.utils import str_to_bool

pytestmark = pytest.mark.pre_merge


def test_str_to_bool():
    """Test the str_to_bool function with various inputs."""

    # Test string inputs
    assert str_to_bool("true")
    assert not str_to_bool("false")
    assert str_to_bool("TRUE")
    assert not str_to_bool("FALSE")
    assert str_to_bool("True")
    assert not str_to_bool("False")

    # Test boolean inputs
    assert str_to_bool(True)
    assert not str_to_bool(False)

    # Test other inputs (should return False)
    assert not str_to_bool("random")
    assert not str_to_bool("")
    assert not str_to_bool(0)
    assert not str_to_bool(1)
    assert not str_to_bool(None)

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

import pytest

pytestmark = pytest.mark.pre_merge


def test_bindings_install():
    # Verify python bindings to rust can be imported
    import dynamo.runtime as tdr

    # Placeholder to avoid unused import errors or removal by linters
    assert tdr


def test_version():
    from dynamo._core import __version__

    assert __version__[0].isdigit()  # semver should start with a digit

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


from dynamo._core import get_reasoning_parser_names, get_tool_parser_names


def test_get_tool_parser_names():
    parsers = get_tool_parser_names()
    # Just make sure it's not None and has some parsers
    # No Need to update this test when adding a new parser everytime
    assert parsers is not None
    assert len(parsers) > 0


def test_get_reasoning_parser_names():
    parsers = get_reasoning_parser_names()
    # Just make sure it's not None and has some parsers
    # No Need to update this test when adding a new parser everytime
    assert parsers is not None
    assert len(parsers) > 0

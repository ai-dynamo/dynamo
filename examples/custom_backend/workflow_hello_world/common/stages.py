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

"""Transport-neutral stage behavior shared by both implementations."""

from collections.abc import Mapping
from typing import Any


class HelloStage:
    """Produce the first half of the fixed response."""

    async def run(self, request: Mapping[str, Any]) -> str:
        del request
        return "Hello, "


class WorldStage:
    """Produce the second half of the fixed response."""

    async def run(self, request: Mapping[str, Any]) -> str:
        del request
        return "World!"


class MergeStage:
    """Join the two stage values."""

    async def run(self, hello: str, world: str) -> str:
        return f"{hello}{world}"

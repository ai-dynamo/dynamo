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

import asyncio
import sys
from pathlib import Path

import uvloop

# Add the project root to the Python path
project_root = str(Path(__file__).parents[1])  # Go up to trtllm directory
if project_root not in sys.path:
    sys.path.append(project_root)

from common.parser import parse_tensorrt_llm_args  # noqa: E402

from .worker import trtllm_worker  # noqa: E402

if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()
    asyncio.run(trtllm_worker(engine_config))

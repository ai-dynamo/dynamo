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

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name == "wheel":
            bin_path = os.getenv("DYNAMO_BIN_PATH", "target/release")
            build_data["force_include"] = {
                f"{bin_path}/dynamo-run": "dynamo/sdk/cli/bin/dynamo-run",
                f"{bin_path}/llmctl": "dynamo/sdk/cli/bin/llmctl",
                f"{bin_path}/http": "dynamo/sdk/cli/bin/http",
                f"{bin_path}/metrics": "dynamo/sdk/cli/bin/metrics",
                f"{bin_path}/mock_worker": "dynamo/sdk/cli/bin/mock_worker",
            }

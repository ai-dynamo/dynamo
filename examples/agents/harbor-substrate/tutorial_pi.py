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

"""Keep the runner's model credential out of Harbor configurations and exec logs."""

import os

from dynamo_pi import DynamoPi


class TutorialPi(DynamoPi):
    def _dynamo_env(self) -> dict[str, str]:
        environment = super()._dynamo_env()
        environment.pop("DYNAMO_API_KEY", None)
        return environment

    async def run(self, instruction, environment, context) -> None:
        # Harbor logs explicit command env. Scoped env is merged by the remote
        # environment after that logging boundary and cleared when run exits.
        with environment.scoped_exec_env(
            {"DYNAMO_API_KEY": os.environ.get("MODEL_API_KEY", "dynamo-local")}
        ):
            await super().run(instruction, environment, context)

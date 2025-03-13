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

import subprocess
from dynamo.sdk import service

@service(
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Frontend:
    def __init__(self):
        subprocess.run(["http"]),
        subprocess.run(["llmctl", "http", "add", "chat-models", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "dynamo-init.vllm.generate"])
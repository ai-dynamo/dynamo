#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# list models
echo "\n\n### Listing models"
curl http://localhost:8000/v1/models

# create completion
echo "\n\n### Creating completions"
curl -X POST http://localhost:8000/v1/chat/completions \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "model": "mock_model",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      }
    ],
    "max_tokens": 64,
    "stream": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "top_k": 5
  }'
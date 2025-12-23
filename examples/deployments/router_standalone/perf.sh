#/bin/bash
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

model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
type=chat
endpoint=/v1/chat/completions
port=8000

isl=3000
osl=100

concurrency=25
num_requests=100
num_unique_prompts=10

seed=42

aiperf profile \
  --model ${model} \
  --tokenizer ${model} \
  --endpoint-type ${type} \
  --endpoint ${endpoint} \
  --streaming \
  --url http://localhost:${port} \
  --synthetic-input-tokens-mean ${isl} \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean ${osl} \
  --output-tokens-stddev 0 \
  --extra-inputs max_tokens:${osl} \
  --extra-inputs min_tokens:${osl} \
  --extra-inputs ignore_eos:true \
  --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
  --concurrency ${concurrency} \
  --request-count ${num_requests} \
  --num-dataset-entries ${num_unique_prompts} \
  --random-seed ${seed} \
  -H 'Authorization: Bearer NOT USED' \
  -H 'Accept: text/event-stream'

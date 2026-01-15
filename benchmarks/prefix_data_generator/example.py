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

import json
import tempfile

import requests
from prefix_data_generator.hasher import hashes_to_texts
from prefix_data_generator.synthesizer import Synthesizer

# download the mooncake trace file
mooncake_trace_permalink = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/f09c501b2a5d73e4d60cdeb612d7d0d54e1ec228/mooncake_trace.jsonl"
with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w+b") as tmp_file:
    response = requests.get(mooncake_trace_permalink)
    tmp_file.write(response.content)
    trace_file = tmp_file.name


# create the synthesizer
synthesizer = Synthesizer(
    dataset_file=trace_file,
    block_size=512,  # it has to be this, as determined by the mooncake trace
    speedup_ratio=2,  # the requests will be sent twice as fast
    prefix_root_multiplier=4,  # will generate 4 separate prefix roots
    prefix_len_multiplier=4,  # prefix lengths 4 times as long
    prompt_len_multiplier=0.5,  # shorten prompt lengths to make prefix ratio even larger
)

# generate requests
requests_synth = synthesizer.synthesize_requests(
    num_requests=100,
    max_isl=(
        16384 - 1000
    ),  # this is what most model defaults to, leaving some room for outputs
)

# convert the hashes into random texts (lorem ipsum), respecting the prefix structure
tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
input_texts = hashes_to_texts(
    tokenizer=tokenizer,
    hash_ids_list=[req["hash_ids"] for req in requests_synth],
    input_lengths=[req["input_length"] for req in requests_synth],
    block_size=512,
)

for i, req in enumerate(requests_synth):
    req["input_text"] = input_texts[i]
    del req["hash_ids"]

output_file = "synthesized_requests.jsonl"
with open("synthesized_requests.jsonl", "w") as f:
    for req in requests_synth:
        f.write(json.dumps(req) + "\n")

print(f"Saved {len(requests_synth)} requests to {output_file}")

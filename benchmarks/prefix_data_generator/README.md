<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License. -->

## Trace File Format

The following tools help analyze and synthesize new data based on the [mooncake trace file format](https://github.com/kvcache-ai/Mooncake/blob/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl). In this format, the first few lines would look like this, for example:

```
{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
{"timestamp": 0, "input_length": 7319, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}
{"timestamp": 3052, "input_length": 7234, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]}
{"timestamp": 3052, "input_length": 2287, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]}
```

**Hash ID Generation:** Each new hash ID is the next consecutive integer after the last one used. Two `hash_ids` sharing the same integers represents the prefix overlap. To generate hash IDs from a list of texts, use `texts_to_hashes` from `aiperf.dataset.synthesis.rolling_hasher`.

**Timestamp:** The arrival time (in milliseconds) of the request since the first request, which can be the same for multiple requests arriving simultaneously.

**Block Size and Hash IDs:** In this example, the `block_size` (the page size of the KV cache) is assumed to be 512. The length of the `hash_ids` array equals `input_length // block_size`.

## Prefix Analyzer

The Prefix Analyzer provides statistics on a trace file, such as Input Sequence Length (ISL), Output Sequence Length (OSL), and theoretical cache hit rate.
It is useful for understanding the structure and reuse patterns in your dataset.

```bash
datagen analyze --input-file <path_to_trace.jsonl> --block-size <block_size>
```

- `--input-file`: Path to your trace file in jsonl format (default: `mooncake_trace.jsonl`)
- `--block-size`: Block size for prefix calculation (default: 512)

The script will print out summary statistics for ISL, OSL, user prompt lengths, and the theoretical cache hit rate (assuming an infinite cache).

## Synthesizer

The Synthesizer goes a step further:
It builds a prefix tree from the original trace file, extracts prefix statistics, and generates a new synthetic dataset based on these statistics.
You can control various aspects of the synthetic data generation with tunable knobs, such as request rate, context/prompt length multipliers, and the number of tree copies.

This is useful for generating large, realistic synthetic traces for benchmarking or simulation, while preserving the structural properties of the original dataset.

### How to run

```bash
datagen synthesize --input-file <path_to_trace.jsonl> --num-requests <N> [other options...]
```

**Options:**
- `--input-file`: Path to the input trace file (default: `mooncake_trace.jsonl`)
- `--num-requests`: Number of requests to synthesize (default: 100000)
- `--speedup-ratio`: Factor to speed up request intervals. It effectively divides the synthetic timestamps by this value (default: 1)
- `--prefix-len-multiplier`: Multiplier for prefix lengths (default: 1.0)
- `--prefix-root-multiplier`: Number of times to replicate the core radix tree (default: 1)
- `--prompt-len-multiplier`: Multiplier for leaf path lengths (default: 1.0, use <1 for shorter prompts)
- `--max-isl`: Maximum input sequence length to include in output (default: None, no filtering)
- `--block-size`: Block size for prefilling and decoding (default: 512)
- `--output-file`: Path to the output file (default: auto-generated from input file and options)

### Example

Say we only have these hash lists:

```
[0, 1, 2, (3)]
[0, 1]
[0, 1, 2]
[0, (4), (5)]
```

First, we identify the "core prefix nodes" as [0, 1, 2] since they are visited more than once. The nodes [3, 4, 5] would be considered "user prompts" as they only appear once (noted in brackets).

If we set the `prefix-len-multiplier` to 2, then the core prefix branches will be stretched, effectively giving:

```
[0, 1, 2, 3, 4, 5, (6)]
[0, 1, 2, 3]
[0, 1, 2, 3, 4, 5]
[0, 1, (7), (8)]
```


Note that the "prompt branches" are not stretched by `prefix-len-multiplier`. They can be separately modified by applying `prompt-len-multiplier`.

Now, if we set `prefix-root-multiplier` to 2, then each row will have a 50 percent chance of being incremented by a large integer, so that they will be effectively separated into a new radix tree, which matches the statistics of the original one, but having completely different roots.

For example, if rows 2 and 4 are offseted, then we would get:

```
[0, 1, 2, 3, 4, 5, (6)]
[10, 11, 12, 13]
[0, 1, 2, 3, 4, 5]
[10, 11, (14), (15)]
```

### Implementation details

The generation algorithm, simplified, is as follows

- Store the hash ids in a directed tree structure (prefix tree)
- Each directed edge `weight` indicates how many times the edge is traversed, which is needed to compute transition probabilities.
- Contract unary paths (chains) in the tree so that it is in a radix-tree form, meaning every node that is the only child will be contracted with the parent. As a consequence, each node need to store an attribute `length` to indicate the compressed length (1 if no compression). The depth multiplier scales this compressed length (rounded to the nearest integer), effectively increasing the length of each radix node.
- Identify every leaf node that is visited only once, and prune them from the tree, as they are highly likely not part of the core radix tree. In other words, we do not need to store nodes that are part of the actual user prompts.
- At this stage, each node will have (possibly zero) transition probabilities to a child prefix node, to a "user prompt" node, and to a "termination" node. Use these probabilities to sample a path in the core radix tree, the append the path with new hash ids corresponding to a user prompt of length sampled from the dataset. The width multiplier effectively duplicates the entire radix tree the specified number of times, each with a new set of hash ids, creating more diverse request patterns.

## Testing

To test for "correctness", or faithfulness to the original trace statistics, one can run
```
python -m benchmarks.data_utils.synthesizer \
--input-file mooncake_trace.jsonl \
--num-requests 500000 \
```
and compare the synthetic ISL statistics (mean, median, std) to the original ISL statistics, which one can obtain by running
```
python -m benchmarks.data_utils.prefix_analyzer \
--input-file mooncake_trace.jsonl \
```
I find this to be the most "robust" end-to-end test. It is important to sample a large number of requests (e.g., hundreds of thousands) to ensure the statistics are meaningful, due to the law of large numbers. In particular, the mean statistics (such as mean ISL) should be well preserved in the synthetic data. However, the standard deviation statistics—especially for ISL—are not expected to match exactly, since the synthesizer does not capture the correlation between context length and prompt length present in the original data.

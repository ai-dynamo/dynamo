<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Prefix Analyzer

The Prefix Analyzer provides statistics on the original trace file, such as Input Sequence Length (ISL), Output Sequence Length (OSL), and theoretical cache hit rate.
It is useful for understanding the structure and reuse patterns in your dataset.

```bash
python -m benchmarks.data_utils.prefix_analyzer --input-file <path_to_trace.jsonl> --block-size <block_size>
```

- `--input-file`: Path to your trace file in jsonl format (default: `mooncake_trace.jsonl`)
- `--block-size`: Block size for prefix calculation (default: 512)

---

The script will print out summary statistics for ISL, OSL, user prompt lengths, and the theoretical cache hit rate (assuming an infinite cache).

## Synthesizer

The Synthesizer goes a step further:
It builds a prefix tree from the original trace file, extracts prefix statistics, and generates a new synthetic dataset based on these statistics.
You can control various aspects of the synthetic data generation with tunable knobs, such as request rate, context/prompt length multipliers, and the number of tree copies.

This is useful for generating large, realistic synthetic traces for benchmarking or simulation, while preserving the structural properties of the original dataset.

### How to run

```bash
python -m benchmarks.data_utils.synthesizer --input-file <path_to_trace.jsonl> --num-requests <N> [other options...]
```

**Options:**
- `--input-file`: Path to the input trace file (default: `mooncake_trace.jsonl`)
- `--num-requests`: Number of requests to synthesize (default: 100000)
- `--speedup-ratio`: Factor to speed up request intervals (default: 1)
- `--depth-multiplier`: Multiplier for prefix lengths (default: 1.0)
- `--width-multiplier`: Number of times to replicate the core radix tree (default: 1)
- `--prompt-len-multiplier`: Multiplier for leaf path lengths (default: 1.0, use <1 for shorter prompts)
- `--max-isl`: Maximum input sequence length to include in output (default: None, no filtering)
- `--block-size`: Block size for prefilling and decoding (default: 512)
- `--output-file`: Path to the output file (default: auto-generated from input file and options)

---

This directory is currently used for generating synthetic data based on the mooncake dataset, but should be easily extendible to any request datasets with (prefix) hash ids, with a current caveat. The synthesizer is designed to work for jsonl files in the "mooncake" trace file format, meaning that the input are increasing integers of block hashes. For now, new block hashes must be the next consecutive integer, otherwise will not work.

If you want to generate these increasing hash ids from a list of texts, you can use the `texts_to_hashes` function in `hasher.py`.

### How it works

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

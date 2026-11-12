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
import math
from collections import Counter
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from prefix_data_generator.graph_utils import (
    CACHE_END,
    END_NODE,
    SUPER_ROOT,
    _mark_visited,
    _merge_chains,
    _precompute_transition_cdfs,
    _remove_leaves,
    _verify_tree,
)
from prefix_data_generator.hasher import RollingHasher
from prefix_data_generator.sampler import EmpiricalSampler, sample_from_cdf


def _validate_integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class Synthesizer:
    def __init__(
        self,
        dataset_file: str,
        block_size: int = 512,
        speedup_ratio: float = 1.0,
        prefix_root_multiplier: int = 1,
        prefix_len_multiplier: float = 1.0,
        prompt_len_multiplier: float = 1.0,
        osl_multiplier: float = 1.0,
        seed: int = 0,
    ):
        """Load the mooncake dataset and extract core statistics like
        radix-tree structure, ISL, OSL, and request timings.
        Generate synthetic datasets based on these statistics, with tunable knobs,
        e.g. to increase request rate or the ISL.

        A request is broken into two parts: a context and a prompt. A context is
        any block that is (can possibly be) visited more than once, while a prompt
        is considered to be unique and only visited once (user prompt).

        Args:
            dataset_file (str): The mooncake trace file in jsonl format.
            block_size (int, optional): The block size for prefilling and decoding.
                Defaults to 512.
            speedup_ratio (float, optional): For speeding up the request intervals.
                Defaults to 1.
            prefix_len_multiplier (float, optional): For every node in the core radix-tree,
                increase the substring length by this multiplier, and rounded to the nearest
                multiple of the block size. In other words, shared prefix prompts will be
                expanded by this factor. Defaults to 1.
            prefix_root_multiplier (int, optional): Number of times to replicate the core radix tree.
                Defaults to 1.
            prompt_len_multiplier (float, optional): Multiplies the leaf path lengths by this factor
                (rounded to integers). Use values < 1 to generate shorter prompts. Defaults to 1.
                Note this does not affect the lengths of the core context prompts.
            osl_multiplier (float, optional): Multiplies output sequence lengths by this factor.
                Defaults to 1.
            seed (int, optional): Seed for all sampling. Defaults to 0.

        Hash IDs are normalized into a synthetic namespace; do not combine them
        with the original trace. Length multipliers scale whole block counts;
        a terminal partial block retains its sampled token remainder.
        """
        self.block_size = block_size
        self.num_copies = prefix_root_multiplier
        self.speedup_ratio = float(speedup_ratio)
        self.prefix_len_multiplier = float(prefix_len_multiplier)
        self.prompt_len_multiplier = float(prompt_len_multiplier)
        self.osl_multiplier = float(osl_multiplier)

        self.rng = np.random.default_rng(seed)
        for name, value in (
            ("block_size", block_size),
            ("prefix_root_multiplier", prefix_root_multiplier),
        ):
            _validate_integer(name, value, minimum=1)
        for name, value in (
            ("speedup_ratio", self.speedup_ratio),
            ("prefix_len_multiplier", self.prefix_len_multiplier),
            ("prompt_len_multiplier", self.prompt_len_multiplier),
            ("osl_multiplier", self.osl_multiplier),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        # extract data from json file
        with open(dataset_file, "r") as f:
            hash_ids_list = []
            timestamps = []
            input_lens = []
            output_lens = []
            for line_number, line in enumerate(f, 1):
                data = json.loads(line)
                hash_ids = data["hash_ids"]
                input_len = data["input_length"]
                output_len = data["output_length"]
                _validate_integer(f"line {line_number}: input_length", input_len, 1)
                _validate_integer(f"line {line_number}: output_length", output_len, 0)
                if not isinstance(hash_ids, list) or not hash_ids:
                    raise ValueError(f"line {line_number}: hash_ids must be nonempty")
                for hash_id in hash_ids:
                    _validate_integer(f"line {line_number}: hash_id", hash_id, 0)
                if not 0 < input_len - (len(hash_ids) - 1) * block_size <= block_size:
                    raise ValueError(
                        f"line {line_number}: hash count does not match input_length"
                    )
                timestamp = float(data["timestamp"])
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError(
                        f"line {line_number}: timestamp must be finite and nonnegative"
                    )
                if timestamps and timestamp < timestamps[-1]:
                    raise ValueError(
                        f"line {line_number}: timestamps must be nondecreasing"
                    )
                hash_ids_list.append(hash_ids)
                timestamps.append(timestamp)
                input_lens.append(input_len)
                output_lens.append(output_len)
        if not hash_ids_list:
            raise ValueError("trace must contain at least one request")

        # Normalize hash_ids to consecutive integers starting from 0
        hasher = RollingHasher()
        hash_ids_list = [
            hasher.hash_token_blocks([(h,) for h in hash_ids])
            for hash_ids in hash_ids_list
        ]

        # represent prefix-tree as directed graph
        self.G = nx.DiGraph()
        max_hash_id = SUPER_ROOT
        num_paths = 0

        self.G.add_node(-1, end=0)
        for hash_ids, input_len in zip(hash_ids_list, input_lens):
            num_paths += 1
            for i in range(len(hash_ids)):
                u = hash_ids[i - 1] if i > 0 else SUPER_ROOT
                v = hash_ids[i]
                max_hash_id = max(v, max_hash_id)

                if v in self.G:
                    self.G.nodes[v]["visited"] += 1
                else:
                    self.G.add_node(v, visited=1, end=0)

                if self.G.has_edge(u, v):
                    self.G[u][v]["weight"] += 1
                else:
                    self.G.add_edge(u, v, weight=1)

            terminal = self.G.nodes[v]
            terminal["end"] += 1
            terminal.setdefault("terminal_remainders", []).append(
                input_len - (len(hash_ids) - 1) * block_size
            )

        self.G.nodes[SUPER_ROOT]["visited"] = num_paths
        self.max_hash_id = max_hash_id

        _verify_tree(self.G)
        _mark_visited(self.G)
        self.G = _merge_chains(self.G)  # make graph radix-like
        self.G, leaves_lens = _remove_leaves(self.G)

        # Apply prompt_len_multiplier to leaves_lens
        if self.prompt_len_multiplier != 1:
            leaves_lens = [
                max(1, round(length * self.prompt_len_multiplier))
                for length in leaves_lens
            ]

        self.leaves_lens_sampler = EmpiricalSampler(leaves_lens, rng=self.rng)
        self._relabel_nodes()
        self.G = _precompute_transition_cdfs(self.G)
        for node, attrs in self.G.nodes(data=True):
            if node != SUPER_ROOT and attrs["end"]:
                attrs["terminal_sampler"] = EmpiricalSampler(
                    attrs.pop("terminal_remainders"), rng=self.rng
                )

        # get statistics of timing, request counts, ISL, and OSL
        request_counts = list(Counter(timestamps).values())
        self.request_counts_sampler = EmpiricalSampler(request_counts, rng=self.rng)
        timedeltas = np.diff(timestamps)
        timedeltas = timedeltas[timedeltas > 0]
        self.timedeltas_sampler = EmpiricalSampler(timedeltas, rng=self.rng)
        input_lens_mod = np.array(
            [
                input_len - (len(hash_ids) - 1) * block_size
                for input_len, hash_ids in zip(input_lens, hash_ids_list)
            ]
        )
        self.input_lens_mod_sampler = EmpiricalSampler(input_lens_mod, rng=self.rng)
        self.output_lens_sampler = EmpiricalSampler(output_lens, rng=self.rng)

    def _relabel_nodes(self) -> None:
        if self.prefix_len_multiplier == 1:
            return
        # Reserve disjoint ranges for expanded chains independently of rounding.
        stride = max(1, math.ceil(self.prefix_len_multiplier))
        for node, attrs in self.G.nodes(data=True):
            if node != SUPER_ROOT:
                attrs["length"] = max(
                    1, round(attrs["length"] * self.prefix_len_multiplier)
                )
        if stride > 1:
            mapping = {node: node * stride + stride - 1 for node in self.G if node >= 0}
            self.G = nx.relabel_nodes(self.G, mapping)
            self.max_hash_id = (self.max_hash_id + 1) * stride - 1

    def _synthesize_leaf_path(self) -> list[int]:
        # Sample the leaf path length
        leaf_length = self.leaves_lens_sampler.sample()

        # Generate new nodes starting from max_hash_id + 1
        path = [int(self.max_hash_id + 1 + i) for i in range(leaf_length)]

        # Update max_hash_id
        self.max_hash_id += leaf_length

        return path

    def synthesize_path(self) -> tuple[list[int], bool, int]:
        """
        Synthesizes a path through the core radix tree, optionally appending a unique user prompt (leaf path).

        Returns:
            tuple:
                - list[int]: The full path as a list of hash_ids. This consists of the cached (core) hash_ids,
                  with new unique hash_ids appended at the end if a leaf path is included.
                - bool: Whether the path contains a leaf path (i.e., new unique hash_ids were appended).
                - int: Shared context token count, including any terminal partial block.
        """
        # Start from root node (-1)
        current_node = SUPER_ROOT
        path: list[int] = []
        context_len = 0

        # Continue until we reach a node with no outgoing edges
        while True:
            # Use precomputed CDFs for efficient sampling
            next_node = sample_from_cdf(
                self.G.nodes[current_node]["out_nodes"],
                self.G.nodes[current_node]["out_cdf"],
                self.rng,
            )

            # end early
            # break and start sampling unique user prompt
            if next_node == CACHE_END:
                break
            # break and don't sample leaf
            if next_node == END_NODE:
                remainder = self.G.nodes[current_node]["terminal_sampler"].sample()
                return path, False, int(context_len - self.block_size + remainder)

            # otherwise continue down prefix tree

            # Get the length of the contracted path
            length = self.G.nodes[next_node]["length"]
            context_len += length * self.block_size

            # Add all intermediate nodes
            for i in range(length):
                path.append(int(next_node - (length - 1) + i))

            current_node = next_node

        unique_user_prompt = self._synthesize_leaf_path()
        # Append a leaf path at the end
        return path + unique_user_prompt, True, context_len

    def synthesize_requests(
        self,
        num_requests: int,
        max_isl: Optional[int] = None,
        min_isl: Optional[int] = None,
        min_osl: Optional[int] = None,
        max_osl: Optional[int] = None,
        max_rejections: int = 10000,
    ) -> list[dict[str, Any]]:
        """Generate requests; fail after max_rejections consecutive ISL rejects."""
        _validate_integer("num_requests", num_requests, 0)
        _validate_integer("max_rejections", max_rejections, 1)
        for name, value, minimum in (
            ("min_isl", min_isl, 1),
            ("max_isl", max_isl, 1),
            ("min_osl", min_osl, 0),
            ("max_osl", max_osl, 0),
        ):
            if value is not None:
                _validate_integer(name, value, minimum)
        for name, lower, upper in (
            ("ISL", min_isl, max_isl),
            ("OSL", min_osl, max_osl),
        ):
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{name} minimum must not exceed maximum")
        timestamp = 0.0
        rejections = 0

        requests: list[dict[str, Any]] = []
        request_id = 0

        while request_id < num_requests:
            requests_per_interval = self.request_counts_sampler.sample()

            for _ in range(requests_per_interval):
                path, leaf_flag, context_len = self.synthesize_path()
                if leaf_flag:
                    input_len = (
                        len(path) - 1
                    ) * self.block_size + self.input_lens_mod_sampler.sample()
                else:
                    input_len = context_len
                output_len = int(
                    self.output_lens_sampler.sample() * self.osl_multiplier
                )

                if (max_isl is not None and input_len > max_isl) or (
                    min_isl is not None and input_len < min_isl
                ):
                    rejections += 1
                    if rejections >= max_rejections:
                        raise ValueError(
                            f"ISL filter rejected {rejections} consecutive requests "
                            f"(min_isl={min_isl}, max_isl={max_isl}); "
                            "relax bounds or increase max_rejections"
                        )
                    continue
                rejections = 0

                # Apply clipping for OSL (not filtering)
                if min_osl is not None and output_len < min_osl:
                    output_len = min_osl
                if max_osl is not None and output_len > max_osl:
                    output_len = max_osl
                requests.append(
                    {
                        "timestamp": timestamp,
                        "input_length": int(input_len),
                        "output_length": int(output_len),
                        "hash_ids": path,
                        "context_len": int(context_len),
                        "unique_user_prompt_len": int(input_len - context_len),
                    }
                )
                request_id += 1
                if request_id >= num_requests:
                    break

            timestamp += self.timedeltas_sampler.sample() / self.speedup_ratio

        # Adjust hash_ids if num_copies > 1
        if self.num_copies > 1:
            for request in requests:
                copy_index = int(self.rng.integers(self.num_copies))
                request["hash_ids"] = [
                    hash_id * self.num_copies + copy_index
                    for hash_id in request["hash_ids"]
                ]

        return requests

    def __repr__(self) -> str:
        path_lengths = nx.single_source_shortest_path_length(self.G, -1)
        core_radix_tree_size = len(self.G) - 1
        core_radix_tree_depth = max(path_lengths.values()) if path_lengths else 0

        rep = "MooncakeSynth("
        rep += f"core_radix_tree_size={core_radix_tree_size}, "
        rep += f"core_radix_tree_depth={core_radix_tree_depth}, "
        rep += f"block_size={self.block_size})"

        children = list(self.G.successors(-1))
        data = {
            "Child Node": children,
            "Visited Count": [self.G.nodes[child]["visited"] for child in children],
            "Length": [self.G.nodes[child].get("length", "N/A") for child in children],
        }
        df = pd.DataFrame(data)
        df = df[df["Visited Count"] >= 5]
        df = df.sort_values("Visited Count", ascending=False)
        grouped = df.groupby("Length", sort=True)

        rep += "\nRoot nodes (grouped by length, visited count ≥ 5):\n"
        for length, group in grouped:
            nodes = group["Child Node"].tolist()
            visit_counts = group["Visited Count"].tolist()
            rep += f"\nNodes: {nodes}, Path Length: {length}, Visited Counts: {visit_counts}"

        return rep


def main():
    import argparse
    from pathlib import Path

    from prefix_data_generator.logging_utils import calculate_and_print_statistics

    parser = argparse.ArgumentParser(description="Synthesize Mooncake-Esque dataset")
    parser.add_argument(
        "--input-file",
        default="mooncake_trace.jsonl",
        type=str,
        help="Path to the input Mooncake JSONL file",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=int(1e5),
        help="Number of requests to synthesize (default: 100000)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=1,
        help="Factor to speed up request intervals (default: 1)",
    )
    parser.add_argument(
        "--prefix-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for prefix lengths (default: 1.0)",
    )
    parser.add_argument(
        "--prefix-root-multiplier",
        type=int,
        default=1,
        help="Number of times to replicate the core radix tree (default: 1)",
    )
    parser.add_argument(
        "--prompt-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for leaf path lengths (default: 1.0, use <1 for shorter prompts)",
    )
    parser.add_argument(
        "--osl-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for output sequence lengths (default: 1.0)",
    )
    parser.add_argument(
        "--max-isl",
        type=int,
        default=None,
        help="Maximum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--min-isl",
        type=int,
        default=None,
        help="Minimum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--min-osl",
        type=int,
        default=None,
        help="Minimum output sequence length - clips values below this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--max-osl",
        type=int,
        default=None,
        help="Maximum output sequence length - clips values above this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Block size for prefilling and decoding (default: 512)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output file (default: None, no output)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for all sampling (default: 0)"
    )
    parser.add_argument(
        "--max-rejections",
        type=int,
        default=10000,
        help="Maximum consecutive ISL rejections (default: 10000)",
    )
    args = parser.parse_args()

    dataset_file = Path(args.input_file).resolve()

    if args.output_file is None:
        suffix_parts = [
            f"{dataset_file.stem}_synth",
            f"{args.prefix_len_multiplier:g}x{args.prefix_root_multiplier}+{args.prompt_len_multiplier}",
            f"speedup{args.speedup_ratio}",
        ]
        if args.max_isl is not None:
            suffix_parts.append(f"maxisl{args.max_isl}")
        if args.min_isl is not None:
            suffix_parts.append(f"minisl{args.min_isl}")
        if args.min_osl is not None:
            suffix_parts.append(f"minosl{args.min_osl}")
        if args.max_osl is not None:
            suffix_parts.append(f"maxosl{args.max_osl}")
        if args.osl_multiplier != 1.0:
            suffix_parts.append(f"oslx{args.osl_multiplier:.1f}")
        output_file = dataset_file.with_stem("_".join(suffix_parts))
    else:
        output_file = Path(args.output_file).resolve()

    print("learning from dataset...", flush=True)
    synthesizer = Synthesizer(
        str(dataset_file),
        block_size=args.block_size,
        speedup_ratio=args.speedup_ratio,
        prefix_len_multiplier=args.prefix_len_multiplier,
        prefix_root_multiplier=args.prefix_root_multiplier,
        prompt_len_multiplier=args.prompt_len_multiplier,
        osl_multiplier=args.osl_multiplier,
        seed=args.seed,
    )

    print("synthesizing requests...", flush=True)
    requests = synthesizer.synthesize_requests(
        args.num_requests,
        max_isl=args.max_isl,
        min_isl=args.min_isl,
        min_osl=args.min_osl,
        max_osl=args.max_osl,
        max_rejections=args.max_rejections,
    )
    print(f"synthesized {len(requests)} requests")

    # Print statistics in a single table with metrics as rows and statistics as columns
    print("\n###### Synthesized Statistics ######")

    # Extract all values first
    metrics = {
        "Input Length": [req["input_length"] for req in requests],
        "Context Length": [req["context_len"] for req in requests],
        "Unique Prompt Length": [req["unique_user_prompt_len"] for req in requests],
        "Output Length": [req["output_length"] for req in requests],
    }

    # Calculate statistics for each metric
    if requests:
        calculate_and_print_statistics(metrics)

    with open(output_file, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    print(f"synthetic dataset saved at {Path(output_file).resolve()}")


if __name__ == "__main__":
    main()

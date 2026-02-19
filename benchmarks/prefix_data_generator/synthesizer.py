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
from collections import Counter
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from aiperf.dataset.synthesis import RollingHasher
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
from prefix_data_generator.sampler import EmpiricalSampler, sample_from_cdf


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
            speedup_ratio (int, optional): For speeding up the request intervals.
                Defaults to 1.
            context_len_multiplier (float, optional): For every node in the core radix-tree,
                increase the substring length by this multiplier, and rounded to the nearest
                multiple of the block size. In other words, shared prefix prompts will be
                expanded by this factor. Defaults to 1.
            num_copies (int, optional): Number of times to replicate the core radix tree.
                Defaults to 1.
            prompt_len_multiplier (float, optional): Multiplies the leaf path lengths by this factor
                (rounded to integers). Use values < 1 to generate shorter prompts. Defaults to 1.
                Note this does not affect the lengths of the core context prompts.
            osl_multiplier (float, optional): Multiplies output sequence lengths by this factor.
                Defaults to 1.

        NOTE: currently may only work for the mooncake trace file,
            as it assumes consecutive integers

        NOTE: If the context_len_multiplier is not one, then the synthetic data
            cannot be mixed and matched with the original trace file,
            as the hash ids will be relabeled.
        """
        self.block_size = block_size
        self.num_copies = prefix_root_multiplier
        self.speedup_ratio = float(speedup_ratio)
        self.prefix_len_multiplier = float(prefix_len_multiplier)
        self.prompt_len_multiplier = float(prompt_len_multiplier)
        self.osl_multiplier = float(osl_multiplier)

        # assert correct arg bounds
        assert (
            isinstance(self.num_copies, int) and self.num_copies >= 1
        ), "num_copies must be an integer greater than or equal to 1"
        assert (
            isinstance(self.speedup_ratio, float) and self.speedup_ratio > 0
        ), "speedup_ratio must be a positive float"
        assert (
            isinstance(self.prefix_len_multiplier, float)
            and self.prefix_len_multiplier > 0
        ), "prefix_len_multiplier must be a positive float"
        assert (
            isinstance(self.prompt_len_multiplier, float)
            and self.prompt_len_multiplier > 0
        ), "prompt_len_multiplier must be a positive float"

        # extract data from json file
        with open(dataset_file, "r") as f:
            hash_ids_list = []
            timestamps = []
            input_lens = []
            output_lens = []
            for line in f:
                data = json.loads(line)
                hash_ids_list.append(data["hash_ids"])
                timestamps.append(int(data["timestamp"]))
                input_lens.append(int(data["input_length"]))
                output_lens.append(int(data["output_length"]))

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
        for hash_ids in hash_ids_list:
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

            self.G.nodes[v]["end"] += 1

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

        self.leaves_lens_sampler = EmpiricalSampler(leaves_lens)
        self._relabel_nodes()
        self.G = _precompute_transition_cdfs(self.G)

        # get statistics of timing, request counts, ISL, and OSL
        request_counts = list(Counter(timestamps).values())
        self.request_counts_sampler = EmpiricalSampler(request_counts)
        timedeltas = np.diff(timestamps)
        timedeltas = timedeltas[timedeltas > 0]
        self.timedeltas_sampler = EmpiricalSampler(timedeltas)
        input_lens_mod = np.array(
            [
                input_len - (len(hash_ids) - 1) * block_size
                for input_len, hash_ids in zip(input_lens, hash_ids_list)
            ]
        )
        assert np.all(0 < input_lens_mod) and np.all(input_lens_mod <= self.block_size)
        self.input_lens_mod_sampler = EmpiricalSampler(input_lens_mod)
        self.output_lens_sampler = EmpiricalSampler(output_lens)

    def _relabel_nodes(self) -> None:
        # Scale node labels by length multiplier if needed
        if self.prefix_len_multiplier > 1:
            multiplier = int(np.ceil(self.prefix_len_multiplier))

            # Scale length attributes BEFORE relabeling
            for node in self.G.nodes():
                if node >= 0:  # Skip special nodes
                    self.G.nodes[node]["length"] = (
                        self.G.nodes[node]["length"] * multiplier
                    )

            # Create mapping for relabeling, preserving -1 and -2
            mapping = {
                node: (node if node < 0 else node * multiplier + multiplier)
                for node in self.G.nodes()
            }
            self.G = nx.relabel_nodes(self.G, mapping)
            # Update max_hash_id
            self.max_hash_id = multiplier * self.max_hash_id + multiplier

        # Shrink the lengths, but no need to relabel nodes
        elif self.prefix_len_multiplier < 1:
            for node in self.G.nodes():
                self.G.nodes[node]["length"] = max(
                    round(self.G.nodes[node]["length"] * self.prefix_len_multiplier), 1
                )

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
                - int: The context length, defined as the number of cached hash_ids multiplied by block_size.
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
            )

            # end early
            # break and start sampling unique user prompt
            if next_node == CACHE_END:
                break
            # break and don't sample leaf
            if next_node == END_NODE:
                return path, False, context_len

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

    @staticmethod
    def _get_current_rate(
        timestamp_ms: int, rate_schedule: list[tuple[int, float]]
    ) -> float:
        """Return the rate for the given timestamp based on the schedule.

        The rate value is either a multiplier (default mode) or an absolute
        req/s value (constant-rate mode), depending on the caller's context.
        """
        elapsed = 0
        for duration_ms, rate in rate_schedule:
            if timestamp_ms < elapsed + duration_ms:
                return rate
            elapsed += duration_ms
        return rate_schedule[-1][1]

    def _build_request(
        self,
        timestamp: int,
        max_isl: Optional[int],
        min_isl: Optional[int],
        min_osl: Optional[int],
        max_osl: Optional[int],
    ) -> Optional[dict[str, Any]]:
        """Build a single request dict, returning None if ISL filters exclude it."""
        path, leaf_flag, context_len = self.synthesize_path()
        if leaf_flag:
            input_len = (
                len(path) - 1
            ) * self.block_size + self.input_lens_mod_sampler.sample()
        else:
            input_len = len(path) * self.block_size
        output_len = int(self.output_lens_sampler.sample() * self.osl_multiplier)

        if max_isl is not None and input_len > max_isl:
            return None
        if min_isl is not None and input_len < min_isl:
            return None

        if min_osl is not None and output_len < min_osl:
            output_len = min_osl
        if max_osl is not None and output_len > max_osl:
            output_len = max_osl

        return {
            "timestamp": int(timestamp),
            "input_length": int(input_len),
            "output_length": int(output_len),
            "hash_ids": path,
            "context_len": int(context_len),
            "unique_user_prompt_len": int(input_len - context_len),
        }

    def synthesize_requests(
        self,
        num_requests: Optional[int] = None,
        max_isl: Optional[int] = None,
        min_isl: Optional[int] = None,
        min_osl: Optional[int] = None,
        max_osl: Optional[int] = None,
        rate_schedule: Optional[list[tuple[int, float]]] = None,
        constant_rate: bool = False,
    ) -> list[dict[str, Any]]:
        """Synthesize requests with either a fixed count or a time-based rate schedule.

        Args:
            num_requests: Maximum number of requests to generate. Required if
                rate_schedule is not provided.
            max_isl: Filter out requests with input length above this value.
            min_isl: Filter out requests with input length below this value.
            min_osl: Clip output lengths below this value.
            max_osl: Clip output lengths above this value.
            rate_schedule: List of (duration_ms, rate) tuples defining a
                time-varying request rate. When provided, generation stops when
                the schedule duration is exhausted. In default mode each rate
                acts as a speedup multiplier; in constant-rate mode each rate
                is an absolute req/s value.
            constant_rate: When True, interpret rate_schedule values as absolute
                req/s and emit exactly 1 request per timestamp with
                deterministic, evenly-spaced inter-arrival times. Requires
                rate_schedule.
        """
        if num_requests is None and rate_schedule is None:
            raise ValueError(
                "At least one of num_requests or rate_schedule must be provided"
            )
        if constant_rate and rate_schedule is None:
            raise ValueError("constant_rate requires rate_schedule")

        total_duration_ms = (
            sum(duration for duration, _ in rate_schedule)
            if rate_schedule is not None
            else None
        )

        timestamp = 0
        requests: list[dict[str, Any]] = []
        request_id = 0

        while True:
            if num_requests is not None and request_id >= num_requests:
                break
            if total_duration_ms is not None and timestamp >= total_duration_ms:
                break

            if constant_rate:
                # Constant-rate mode: exactly 1 request per timestamp,
                # deterministic inter-arrival time
                rate = self._get_current_rate(timestamp, rate_schedule)
                request = self._build_request(
                    timestamp, max_isl, min_isl, min_osl, max_osl
                )
                if request is not None:
                    requests.append(request)
                    request_id += 1
                timestamp += round(1000 / rate)
            else:
                # Default mode: sample batch size and inter-arrival from trace
                requests_per_interval = self.request_counts_sampler.sample()

                for _ in range(requests_per_interval):
                    request = self._build_request(
                        timestamp, max_isl, min_isl, min_osl, max_osl
                    )
                    if request is not None:
                        requests.append(request)
                        request_id += 1
                    if num_requests is not None and request_id >= num_requests:
                        break

                if rate_schedule is not None:
                    rate = self._get_current_rate(timestamp, rate_schedule)
                else:
                    rate = self.speedup_ratio
                timestamp += round(self.timedeltas_sampler.sample() / rate)

        # Adjust hash_ids if num_copies > 1
        if self.num_copies > 1:
            for request in requests:
                offset = (np.random.randint(0, self.num_copies)) * (
                    self.max_hash_id + 1
                )
                request["hash_ids"] = [
                    int(hash_id + offset) for hash_id in request["hash_ids"]
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
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Number of requests to synthesize (default: 100000 if no rate schedule)",
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
        "--step-duration",
        type=int,
        default=None,
        help="Duration of each rate step in seconds (use with --step-rates)",
    )
    parser.add_argument(
        "--step-rates",
        type=str,
        default=None,
        help="Comma-separated rate values for staircase load pattern (e.g., '1,2,4,2,1'). "
        "Values are multipliers on the base trace rate by default, or absolute req/s with --constant-rate.",
    )
    parser.add_argument(
        "--constant-rate",
        action="store_true",
        default=False,
        help="Interpret --step-rates as absolute req/s and emit evenly-spaced requests "
        "(requires --step-duration and --step-rates)",
    )
    args = parser.parse_args()

    # Validate rate schedule arguments
    has_schedule = args.step_rates is not None or args.step_duration is not None
    if has_schedule:
        if args.step_rates is None or args.step_duration is None:
            parser.error("--step-duration and --step-rates must be used together")
        if args.speedup_ratio != 1:
            parser.error("--speedup-ratio cannot be used with --step-rates")
    if args.constant_rate:
        if not has_schedule:
            parser.error("--constant-rate requires --step-duration and --step-rates")
        if args.speedup_ratio != 1:
            parser.error("--constant-rate cannot be used with --speedup-ratio")

    rate_schedule = None
    if args.step_rates is not None:
        rates = [float(r) for r in args.step_rates.split(",")]
        if any(r <= 0 for r in rates):
            parser.error("All step rates must be positive")
        if args.step_duration <= 0:
            parser.error("--step-duration must be positive")
        duration_ms = args.step_duration * 1000
        rate_schedule = [(duration_ms, r) for r in rates]

    # Default num_requests when no rate schedule is provided
    if args.num_requests is None and rate_schedule is None:
        args.num_requests = int(1e5)

    dataset_file = Path(args.input_file).resolve()

    if args.output_file is None:
        suffix_parts = [
            f"{dataset_file.stem}_synth",
            f"{int(args.prefix_len_multiplier)}x{args.prefix_root_multiplier}+{args.prompt_len_multiplier}",
        ]
        if rate_schedule is not None:
            rates_str = "-".join(f"{r:g}" for _, r in rate_schedule)
            prefix = "constant" if args.constant_rate else "staircase"
            suffix_parts.append(f"{prefix}_{args.step_duration}s_{rates_str}")
        else:
            suffix_parts.append(f"speedup{args.speedup_ratio}")
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
    )

    if rate_schedule is not None:
        total_duration_s = sum(d for d, _ in rate_schedule) / 1000
        mode = "constant req/s" if args.constant_rate else "multiplier"
        print(
            f"Rate schedule ({mode}): {len(rate_schedule)} steps of "
            f"{args.step_duration}s each ({total_duration_s:.0f}s total)"
        )
        print(f"Step rates: {[r for _, r in rate_schedule]}")
    print("synthesizing requests...", flush=True)
    requests = synthesizer.synthesize_requests(
        num_requests=args.num_requests,
        max_isl=args.max_isl,
        min_isl=args.min_isl,
        min_osl=args.min_osl,
        max_osl=args.max_osl,
        rate_schedule=rate_schedule,
        constant_rate=args.constant_rate,
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
    calculate_and_print_statistics(metrics)

    with open(output_file, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    print(f"synthetic dataset saved at {Path(output_file).resolve()}")


if __name__ == "__main__":
    main()

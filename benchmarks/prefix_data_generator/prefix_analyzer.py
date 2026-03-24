# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

from prefix_data_generator.logging_utils import calculate_and_print_statistics


class PrefixAnalyzer:
    """
    A class for analyzing dataset characteristics related to prefixes, hash IDs, and cache hit rates.
    """

    def __init__(self, dataset_path, block_size=1):
        """
        Initialize the analyzer with dataset path and block size.

        Args:
            dataset_path: Path to the JSONL dataset file
            block_size: Size of each block for prefix calculation
        """
        self.dataset_path = dataset_path
        self.block_size = block_size
        self.dataset = self._load_dataset()
        self.hash_counter = self._build_hash_counter()
        self.repeated_hash_ids = {
            (pos, hash_id)
            for (pos, hash_id), count in self.hash_counter.items()
            if count > 1
        }

    def _load_dataset(self) -> list:
        print(f"Loading dataset from {self.dataset_path}...")
        dataset = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        print(f"Dataset loaded: {len(dataset)} examples")
        return dataset

    def _build_hash_counter(self) -> Counter:
        all_hash_positions = []
        for item in self.dataset:
            for pos, hash_id in enumerate(item["hash_ids"]):
                all_hash_positions.append((pos, hash_id))
        counter = Counter(all_hash_positions)
        print(f"Hash counter built: {len(counter)} unique (position, hash_id) pairs")
        return counter

    def analyze(self) -> dict[str, list]:
        """
        Analyze dataset to extract various length metrics and print statistics.

        Returns:
            Tuple of lists: (input_lengths, prefix_lengths, user_prompt_lengths, output_lengths)
        """
        # Extract input and output lengths directly from fields
        input_lengths = [item["input_length"] for item in self.dataset]
        output_lengths = [item["output_length"] for item in self.dataset]

        # Calculate prefix length and user prompt length for each row
        prefix_lengths = []
        user_prompt_lengths = []

        for i, item in enumerate(self.dataset):
            input_len = item["input_length"]
            hash_ids = item["hash_ids"]
            assert len(hash_ids) * self.block_size >= input_len

            # Special case: if all (position, hash_id) pairs in the row are repeated elsewhere
            if all(
                (pos, hash_id) in self.repeated_hash_ids
                for pos, hash_id in enumerate(hash_ids)
            ):
                prefix_len = input_len  # Set prefix length to input length
                user_prompt_len = 0  # Set user prompt length to 0
            else:
                # Count how many (position, hash_id) pairs in this row are repeated elsewhere in the dataset
                repeated_count = sum(
                    1
                    for pos, hash_id in enumerate(hash_ids)
                    if (pos, hash_id) in self.repeated_hash_ids
                )
                prefix_len = repeated_count * self.block_size
                user_prompt_len = input_len - prefix_len

            prefix_lengths.append(prefix_len)
            user_prompt_lengths.append(user_prompt_len)

            # Check if prefix length is greater than input length
            if prefix_len > input_len:
                print(f"WARNING: Line {i}: {json.dumps(item)}")

        cache_hit_rates = self._analyze_cache_hit_rates()

        # Print statistics table
        metrics = {
            "Input Length": input_lengths,
            "Context Length": prefix_lengths,
            "Unique Prompt Length": user_prompt_lengths,
            "Output Length": output_lengths,
            "Theoretical Hit Rates": cache_hit_rates,
        }

        calculate_and_print_statistics(metrics)

        return metrics

    def _analyze_cache_hit_rates(self) -> list[float]:
        """
        Analyze theoretical cache hit rates based on hash ID repetition.

        Assumes that hash IDs are cached as the dataset is iterated through,
        i.e., each hash ID is considered "cached" after its first appearance,
        similar to how KV caching would work in real life.
        Assumes the cache is infinite in size (hence "theoretical"), so no hash IDs are ever evicted.

        Returns:
            List of cache hit rates for each row in the dataset
        """
        # Set to track all hash IDs we've seen
        seen_hash_ids = set()

        # Store cache hit rates for each row
        cache_hit_rates = []

        for item in self.dataset:
            hash_ids = item["hash_ids"]

            # Skip if there are no hash IDs
            if len(hash_ids) == 0:
                continue

            # Find the first index where the hash ID hasn't been seen before
            first_unseen_idx = len(hash_ids)  # Default if all are seen
            for idx, hash_id in enumerate(hash_ids):
                if hash_id not in seen_hash_ids:
                    first_unseen_idx = idx
                    break

            # Calculate cache hit rate
            cache_hit_rate = first_unseen_idx / len(hash_ids)
            cache_hit_rates.append(cache_hit_rate)

            # Add all hash IDs from this row to the seen set
            seen_hash_ids.update(hash_ids)

        return cache_hit_rates


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze prefix dataset statistics")
    parser.add_argument(
        "--input-file",
        type=str,
        default="mooncake_trace.jsonl",
        help="Path to the input dataset file (default: mooncake_trace.jsonl)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Block size for prefix calculation (default: 512)",
    )
    args = parser.parse_args()

    block_size = args.block_size
    dataset_path = args.input_file

    print(f"Analyzing dataset: {dataset_path}")
    print(f"Using block size: {block_size}")
    print()

    # Create analyzer instance
    analyzer = PrefixAnalyzer(dataset_path, block_size=block_size)
    analyzer.analyze()


if __name__ == "__main__":
    main()

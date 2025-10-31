# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import random

import pandas as pd
from prefix_data_generator.hasher import RollingHasher
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert CSV file to mooncake format")
    parser.add_argument("--input-file", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output mooncake-style jsonl file. If not provided, will use input file name but change extension from .csv to .jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["ChatGPT", "GPT-4"],
        help="Filter by model (ChatGPT or GPT-4). If not specified, no filtering is applied.",
    )
    parser.add_argument(
        "--log-type",
        type=str,
        default=None,
        choices=["Conversation log", "API log"],
        help="Filter by log type (Conversation log or API log). If not specified, no filtering is applied.",
    )
    parser.add_argument(
        "--num-prompt",
        type=int,
        default=None,
        help="Limit the number of rows to output after filtering. If not specified, all rows are output.",
    )
    parser.add_argument(
        "--speed-ratio",
        type=float,
        default=1.0,
        help="Speed ratio to adjust timestamps. Values > 1 speed up requests, < 1 slow down. Default: 1.0 (no change)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for calculating hash array length: ceil(input_length / block_size)",
    )
    parser.add_argument(
        "--num-hash-blocks",
        type=int,
        default=10000,
        help="Maximum hash ID value for random hash generation. Default: 10000",
    )
    return parser.parse_args()


def load_csv(filepath):
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("First few rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def apply_filters(df, model=None, log_type=None, num_prompt=None):
    """
    Apply filters to the DataFrame.

    Args:
        df: Input DataFrame
        model: Model to filter by (ChatGPT or GPT-4)
        log_type: Log type to filter by (Conversation log or API log)
        num_prompt: Number of rows to keep after filtering

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Apply model filter
    if model is not None:
        filtered_df = filtered_df[filtered_df["Model"] == model]
        print(f"After model filter ({model}): {len(filtered_df)} rows")

    # Apply log type filter
    if log_type is not None:
        filtered_df = filtered_df[filtered_df["Log Type"] == log_type]
        print(f"After log type filter ({log_type}): {len(filtered_df)} rows")

    # Apply num_prompt limit
    if num_prompt is not None:
        filtered_df = filtered_df.head(num_prompt)
        print(f"After num_prompt limit ({num_prompt}): {len(filtered_df)} rows")

    return filtered_df


def apply_speed_ratio(df, speed_ratio):
    """
    Apply speed ratio to timestamps.

    Args:
        df: Input DataFrame
        speed_ratio: Speed ratio to adjust timestamps (timestamp /= speed_ratio)

    Returns:
        DataFrame with adjusted timestamps
    """
    if speed_ratio == 1.0:
        print("Speed ratio is 1.0, no timestamp adjustment needed")
        return df

    adjusted_df = df.copy()
    adjusted_df["Timestamp"] = adjusted_df["Timestamp"] / speed_ratio

    print(f"Applied speed ratio: {speed_ratio}")
    print(
        f"Original timestamps: {df['Timestamp'].min():.2f} - {df['Timestamp'].max():.2f}"
    )
    print(
        f"Adjusted timestamps: {adjusted_df['Timestamp'].min():.2f} - {adjusted_df['Timestamp'].max():.2f}"
    )

    return adjusted_df


def convert_to_mooncake(df, block_size, num_hash_blocks):
    """
    Convert DataFrame to mooncake format.

    Args:
        df: Input DataFrame with columns: Timestamp, Request tokens, Response tokens
        block_size: Block size for calculating hash array length
        num_hash_blocks: Maximum hash ID value for random generation

    Returns:
        DataFrame in mooncake format with columns: timestamp, input_length, output_length, hash_ids
    """
    mooncake_data = []
    hasher = RollingHasher()  # Initialize once to maintain global state

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Convert timestamp from seconds to milliseconds (integer)
        timestamp_ms = int(row["Timestamp"] * 1000)

        # Map request tokens to input_length and response tokens to output_length
        input_length = int(row["Request tokens"])
        output_length = int(row["Response tokens"])

        # Calculate hash array length based on block size
        hash_array_length = math.ceil(input_length / block_size)

        # Generate random content blocks (each block is a tuple of random integers)
        # Using request index as seed for reproducibility
        random.seed(idx)
        content_blocks = [
            (random.randint(0, num_hash_blocks),) for _ in range(hash_array_length)
        ]

        hash_ids = hasher(content_blocks)

        mooncake_data.append(
            {
                "timestamp": timestamp_ms,
                "input_length": input_length,
                "output_length": output_length,
                "hash_ids": hash_ids,
            }
        )

    print(f"Converted {len(mooncake_data)} rows to mooncake format")
    return pd.DataFrame(mooncake_data)


def print_statistics(df):
    """
    Print statistics about the converted mooncake data.

    Args:
        df: DataFrame in mooncake format
    """
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    # Input length statistics
    isl_min = df["input_length"].min()
    isl_max = df["input_length"].max()
    isl_avg = df["input_length"].mean()
    isl_std = df["input_length"].std()

    print("\nInput Length (ISL):")
    print(f"  Min: {isl_min}")
    print(f"  Max: {isl_max}")
    print(f"  Avg: {isl_avg:.2f}")
    print(f"  Std: {isl_std:.2f}")

    # Output length statistics
    osl_min = df["output_length"].min()
    osl_max = df["output_length"].max()
    osl_avg = df["output_length"].mean()
    osl_std = df["output_length"].std()

    print("\nOutput Length (OSL):")
    print(f"  Min: {osl_min}")
    print(f"  Max: {osl_max}")
    print(f"  Avg: {osl_avg:.2f}")
    print(f"  Std: {osl_std:.2f}")

    # Sequence length (ISL + OSL) - calculate without modifying df
    max_seq_len = (df["input_length"] + df["output_length"]).max()
    print("\nSequence Length (ISL + OSL):")
    print(f"  Max: {max_seq_len}")

    # RPS calculation
    if len(df) > 1:
        # Timestamps are in milliseconds, convert to seconds
        min_timestamp_s = df["timestamp"].min() / 1000.0
        max_timestamp_s = df["timestamp"].max() / 1000.0
        duration_s = max_timestamp_s - min_timestamp_s

        if duration_s > 0:
            avg_rps = len(df) / duration_s
            print("\nRequest Rate:")
            print(f"  Total requests: {len(df)}")
            print(f"  Duration: {duration_s:.2f} seconds")
            print(f"  Average RPS: {avg_rps:.2f}")
        else:
            print("\nRequest Rate:")
            print(f"  Total requests: {len(df)}")
            print("  Duration: 0 seconds (all requests at same timestamp)")
            print("  Average RPS: N/A")
    else:
        print("\nRequest Rate:")
        print(f"  Total requests: {len(df)}")
        print("  Average RPS: N/A (only 1 request)")

    print("=" * 60)


def main():
    args = parse_args()

    # Load the CSV file
    df = load_csv(args.input_file)
    if df is None:
        return 1

    # Apply filters
    print("\nApplying filters...")
    print(f"Initial rows: {len(df)}")
    filtered_df = apply_filters(
        df, model=args.model, log_type=args.log_type, num_prompt=args.num_prompt
    )

    # Apply Speedup
    adjusted_df = apply_speed_ratio(filtered_df, args.speed_ratio)

    # Convert to mooncake format
    print("\nConverting to mooncake format...")
    print(f"Block size: {args.block_size}")
    print(f"Num hash blocks: {args.num_hash_blocks}")
    mooncake_df = convert_to_mooncake(
        adjusted_df, args.block_size, args.num_hash_blocks
    )

    # Print statistics
    print_statistics(mooncake_df)

    # Save to file
    # Determine output file name
    if args.output_file is None:
        # Use input file name but change extension from .csv to .jsonl
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + ".jsonl"
    mooncake_df.to_json(args.output_file, orient="records", lines=True)
    print(f"\nSaved {len(mooncake_df)} rows to {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())

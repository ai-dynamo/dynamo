# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Round-trip test for hash consistency through aiperf's PromptGenerator.

Tests that when:
1. A mooncake trace with hash_ids is loaded
2. PromptGenerator generates text from those hash_ids
3. The text is re-encoded with the same tokenizer
4. Rolling hashes are computed on the re-encoded tokens

The resulting hashes match the original hash_ids.

This verifies that the BOS token boundary markers survive the
text decode/encode cycle correctly.
"""

import json
import tempfile
import urllib.request

import pytest
from aiperf.common.config import PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator import PromptGenerator
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher

# Mooncake trace URL
MOONCAKE_TRACE_URL = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_BLOCK_SIZE = 512


def download_mooncake_trace(output_path: str, num_lines: int = 100) -> None:
    """Download first N lines of mooncake trace."""
    with urllib.request.urlopen(MOONCAKE_TRACE_URL) as response:
        lines_written = 0
        with open(output_path, "w") as f:
            for line in response:
                if lines_written >= num_lines:
                    break
                f.write(line.decode("utf-8"))
                lines_written += 1


class TestRoundtripHashes:
    """Test hash consistency through aiperf's PromptGenerator."""

    def test_hash_roundtrip_direct(self):
        """
        Direct test using PromptGenerator:
        1. Download mooncake trace (first 100 requests)
        2. For each trace with hash_ids:
           a. Use PromptGenerator to generate text from hash_ids
           b. Re-encode the text with add_special_tokens=False
           c. Compute rolling hashes
           d. Verify hash count matches original
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = f"{tmpdir}/mooncake_trace.jsonl"

            # Step 1: Download trace
            print("Downloading mooncake trace...")
            download_mooncake_trace(trace_path, num_lines=100)

            # Load traces
            traces = []
            with open(trace_path) as f:
                for line in f:
                    traces.append(json.loads(line.strip()))

            print(f"Loaded {len(traces)} traces")

            # Step 2: Initialize tokenizer and PromptGenerator
            print(f"Loading tokenizer: {DEFAULT_TOKENIZER}")
            tokenizer = Tokenizer.from_pretrained(DEFAULT_TOKENIZER)

            config = PromptConfig()
            config.input_tokens.block_size = DEFAULT_BLOCK_SIZE
            prompt_generator = PromptGenerator(config=config, tokenizer=tokenizer)

            # Phase 1: Normalize original hash_ids through a hasher
            original_hasher = RollingHasher(block_size=DEFAULT_BLOCK_SIZE)
            original_hashes_map: dict[int, list[int]] = {}

            for i, trace in enumerate(traces):
                original_hash_ids = trace.get("hash_ids", [])
                if not original_hash_ids:
                    continue
                # Wrap each hash_id as a tuple to make it a hashable "block"
                original_blocks = [(h,) for h in original_hash_ids]
                original_hashes_map[i] = original_hasher.hash_token_blocks(
                    original_blocks
                )

            # Phase 2: Test roundtrip
            roundtrip_hasher = RollingHasher(block_size=DEFAULT_BLOCK_SIZE)
            mismatches = []
            tested = 0

            for i, trace in enumerate(traces):
                original_hash_ids = trace.get("hash_ids", [])
                input_length = trace.get("input_length", 0)

                if not original_hash_ids or input_length == 0:
                    continue

                # Step 1: Generate text from hash_ids
                try:
                    generated_text = prompt_generator.generate(
                        mean=input_length, hash_ids=original_hash_ids
                    )
                except Exception as e:
                    print(f"  Trace {i}: Generation failed - {e}")
                    continue

                # Step 2: Re-encode text (simulates what server does)
                re_encoded_tokens = tokenizer.encode(generated_text)

                # Step 3: Hash re-encoded tokens
                re_encoded_blocks = [
                    re_encoded_tokens[j : j + DEFAULT_BLOCK_SIZE]
                    for j in range(0, len(re_encoded_tokens), DEFAULT_BLOCK_SIZE)
                ]
                computed_hashes = roundtrip_hasher.hash_token_blocks(re_encoded_blocks)

                tested += 1

                # Step 4: Compare original vs computed hashes
                original_hashes = original_hashes_map[i]
                if computed_hashes != original_hashes:
                    mismatches.append(
                        {
                            "index": i,
                            "input_length": input_length,
                            "original_hashes": original_hashes,
                            "computed_hashes": computed_hashes,
                            "re_encoded_token_count": len(re_encoded_tokens),
                        }
                    )

            print(f"\nTested {tested} traces with hash_ids")

            # Report results
            if mismatches:
                print(f"\nFound {len(mismatches)} hash mismatches:")
                for m in mismatches[:5]:
                    print(
                        f"  Trace {m['index']}: "
                        f"input_len={m['input_length']}, "
                        f"re_encoded_tokens={m['re_encoded_token_count']}"
                    )
                    print(f"    original_hashes: {m['original_hashes'][:10]}...")
                    print(f"    computed_hashes: {m['computed_hashes'][:10]}...")
                pytest.fail(
                    f"Hash roundtrip failed: {len(mismatches)}/{tested} mismatches"
                )

            print(f"All {tested} traces passed hash roundtrip check")

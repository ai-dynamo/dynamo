# generate_trace_from_prompts.py

import json
import sys
from typing import List, Dict, Union
from transformers import AutoTokenizer

# === Configuration ===
# BLOCK_SIZE = 512
TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Or any other tokenizer

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# === Hash state (global map and counter) ===
hash_to_int: Dict[int, int] = {}
next_int = 0


def prompt_to_hash_ids(
    prompt: str, block_size: int
) -> Dict[str, Union[int, List[int]]]:
    global next_int

    # Tokenize without special tokens
    encoding = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
    tokens = encoding["input_ids"]
    input_length = len(tokens)
    output_length = 500  # Can be constant or estimated based on task

    # Rolling hash computation
    parent_hash = 0
    hash_ids: List[int] = []

    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        combined = (parent_hash, hash(tuple(block)))
        global_hash = hash(combined)

        if global_hash not in hash_to_int:
            hash_to_int[global_hash] = next_int
            next_int += 1

        hash_ids.append(hash_to_int[global_hash])
        parent_hash = global_hash

    return {
        "timestamp": 0,
        "input_length": input_length,
        "output_length": output_length,
        "hash_ids": hash_ids,
    }


def main(input_path: str, output_path: str, block_size: int):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = data["prompt"]
                trace = prompt_to_hash_ids(prompt, block_size)
                f_out.write(json.dumps(trace) + "\n")
            except Exception as e:
                print(f"Skipping line due to error: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python generate_trace_from_prompts.py <prompts.jsonl> <output.jsonl> <block_size>"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    block_size = int(sys.argv[3])
    main(input_path, output_path, block_size)

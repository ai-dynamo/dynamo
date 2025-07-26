# minrex_prompt_to_hash_ids.py

from transformers import AutoTokenizer
from typing import List, Dict, Union
import numpy as np
import json

# === Constants ===
BLOCK_SIZE = 512
TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or any model on HF

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# === Prompt ===
prompt = (
    "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, "
    "lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, "
    "Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. "
    "You are an intrepid explorer, known for your unparalleled curiosity and courage..."
) * 20  # repeat to simulate a long input

# === Tokenize ===
encoding = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
tokens: List[int] = encoding["input_ids"]
input_length = len(tokens)

# === Compute hash blocks ===
hash_to_int: Dict[int, int] = {}
next_int = 0
hash_ids: List[int] = []
parent_hash = 0

for i in range(0, len(tokens), BLOCK_SIZE):
    block = tokens[i : i + BLOCK_SIZE]
    combined = (parent_hash, hash(tuple(block)))
    global_hash = hash(combined)

    if global_hash not in hash_to_int:
        hash_to_int[global_hash] = next_int
        next_int += 1

    hash_ids.append(hash_to_int[global_hash])
    parent_hash = global_hash

# === Format output ===
trace_entry = {
    "timestamp": 0,
    "input_length": input_length,
    "output_length": 500,  # can be anything
    "hash_ids": hash_ids,
}

# === Print JSONL line ===
print(json.dumps(trace_entry))


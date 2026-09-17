#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare deterministic workloads for the native CPU EC overlap benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # Optional for tokenizer-independent unit tests.
    AutoTokenizer = None

from benchmarks.multimodal.jsonl.generate_images import (
    compute_image_uuid,
    generate_image_pool_base64,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
DEFAULT_OUTPUT_DIR = Path("/dynamo-tmp/data")
DEFAULT_IMAGE_DIR = DEFAULT_OUTPUT_DIR / "ec_h2d_overlap_images_2400x1080_seed42"
SYSTEM_PROMPT_TOKENS = 8000
USER_TEXT = "Describe the newest image and summarize only its visible content."
SYSTEM_CONTEXT = (
    " The attached images may contain objects, text, diagrams, tables, labels,"
    " quantities, and annotations. Consider visual details in context, distinguish"
    " similar elements carefully, and report only information supported by the images."
)


class Tokenizer(Protocol):
    chat_template: str | dict[str, str] | None
    init_kwargs: dict[str, Any]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        ...

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any):
        ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_exact_text(tokenizer: Tokenizer, target_tokens: int) -> str:
    """Return deterministic text with exactly ``target_tokens`` tokenizer tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    unit_tokens = count_tokens(tokenizer, SYSTEM_CONTEXT)
    repeats = max(2, target_tokens // unit_tokens + 2)
    keep = target_tokens
    for _ in range(64):
        ids = tokenizer.encode(SYSTEM_CONTEXT * repeats, add_special_tokens=False)
        while len(ids) < keep:
            repeats *= 2
            ids = tokenizer.encode(SYSTEM_CONTEXT * repeats, add_special_tokens=False)
        text = tokenizer.decode(
            ids[:keep],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        observed = count_tokens(tokenizer, text)
        if observed == target_tokens:
            return text
        keep += target_tokens - observed
    raise RuntimeError(f"could not construct exact {target_tokens}-token text")


def build_chat_template(base_template: str, system_prompt: str) -> str:
    """Prepend the benchmark system message without changing client payloads."""
    prompt_literal = json.dumps(system_prompt, ensure_ascii=False)
    prefix = (
        "{%- if not messages or messages[0]['role'] != 'system' -%}"
        "{%- set messages = [{'role': 'system', 'content': "
        f"{prompt_literal}"
        "}] + messages -%}"
        "{%- endif -%}\n"
    )
    return prefix + base_template


def write_sliding_dataset(
    path: Path,
    image_pool: Sequence[str],
    *,
    num_users: int,
    turns_per_user: int,
    window_size: int,
    images_per_user: int,
    user_text: str,
) -> dict[str, int]:
    """Write one turn-major sliding-window dataset from a shared image pool."""
    required_images = num_users * images_per_user
    if len(image_pool) < required_images:
        raise ValueError(
            f"image pool has {len(image_pool)} entries; {required_images} required"
        )
    if window_size + turns_per_user - 1 > images_per_user:
        raise ValueError("images_per_user cannot cover every sliding window")

    path.parent.mkdir(parents=True, exist_ok=True)
    unique_refs: set[str] = set()
    rows = 0
    with path.open("w", encoding="utf-8") as output:
        for turn_idx in range(turns_per_user):
            for user_idx in range(num_users):
                offset = user_idx * images_per_user + turn_idx
                images = list(image_pool[offset : offset + window_size])
                unique_refs.update(images)
                row = {
                    "session_id": f"user_{user_idx}",
                    "text": user_text,
                    "images": images,
                    "image_uuids": [compute_image_uuid(ref) for ref in images],
                }
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows += 1

    total_slots = rows * window_size
    return {
        "rows": rows,
        "unique_images": len(unique_refs),
        "content_images": len(unique_refs),
        "stripped_images": total_slots - len(unique_refs),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.output_dir / "qwen35_shared_system_8000.txt"
    template_path = args.output_dir / "qwen35_shared_system_8000.jinja"
    manifest_path = args.output_dir / "qwen35_ec_h2d_overlap_manifest.json"
    dataset_paths = {
        5: args.output_dir / "30u_8t_5w_shared8k_base64_uuid_seed42.jsonl",
        10: args.output_dir / "30u_8t_10w_shared8k_base64_uuid_seed42.jsonl",
    }
    artifacts = [prompt_path, template_path, manifest_path, *dataset_paths.values()]
    if not args.force and all(path.is_file() for path in artifacts):
        print(f"All workload artifacts already exist; keeping {manifest_path}")
        return

    if AutoTokenizer is None:
        raise RuntimeError("transformers is required to prepare EC workloads")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_template = tokenizer.chat_template
    if not isinstance(base_template, str):
        raise TypeError("target tokenizer must expose one string chat_template")

    system_prompt = make_exact_text(tokenizer, SYSTEM_PROMPT_TOKENS)
    prompt_path.write_text(system_prompt, encoding="utf-8")
    template = build_chat_template(base_template, system_prompt)
    template_path.write_text(template, encoding="utf-8")

    rendered_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_TEXT}],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=True,
    )

    num_users = 30
    turns_per_user = 8
    max_window_size = max(dataset_paths)
    images_per_user = max_window_size + turns_per_user - 1
    image_pool = generate_image_pool_base64(
        np.random.default_rng(args.seed),
        num_users * images_per_user,
        args.image_dir,
        (2400, 1080),
    )

    datasets: dict[str, dict[str, Any]] = {}
    for window_size, path in dataset_paths.items():
        counts = write_sliding_dataset(
            path,
            image_pool,
            num_users=num_users,
            turns_per_user=turns_per_user,
            window_size=window_size,
            images_per_user=images_per_user,
            user_text=USER_TEXT,
        )
        datasets[str(window_size)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "window_size": window_size,
            **counts,
        }

    manifest = {
        "model": args.model,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "seed": args.seed,
        "system_prompt": {
            "path": str(prompt_path),
            "tokens": count_tokens(tokenizer, system_prompt),
            "sha256": _sha256(prompt_path),
        },
        "chat_template": {
            "path": str(template_path),
            "sha256": _sha256(template_path),
        },
        "user_text": USER_TEXT,
        "user_text_tokens": count_tokens(tokenizer, USER_TEXT),
        "rendered_text_prompt_tokens": len(rendered_ids),
        "images": {
            "directory": str(args.image_dir),
            "count": len(image_pool),
            "width": 2400,
            "height": 1080,
        },
        "datasets": datasets,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

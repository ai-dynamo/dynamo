# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and audit the deterministic Qwen2.5 text workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_PROMPT = "Summarize the benchmark request in one concise sentence."
FILLER = " benchmark"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered_token_count(tokenizer: Any, prompt: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def calibrate_prompt(tokenizer: Any, target_isl: int) -> str:
    for repeat_count in range(target_isl * 2):
        prompt = BASE_PROMPT + FILLER * repeat_count
        token_count = rendered_token_count(tokenizer, prompt)
        if token_count == target_isl:
            return prompt
        if token_count > target_isl:
            break
    raise ValueError(f"could not calibrate an exact rendered ISL of {target_isl}")


def validate_workload(
    dataset_path: Path,
    manifest_path: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as dataset:
        for line in dataset:
            rows.append(json.loads(line))

    expected_rows = int(manifest["request_count"])
    target_isl = int(manifest["target_isl"])
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
    if any(set(row) != {"session_id", "text"} for row in rows):
        raise ValueError("text workload rows must contain only session_id and text")

    prompts = {str(row["text"]) for row in rows}
    session_ids = {str(row["session_id"]) for row in rows}
    if len(prompts) != 1:
        raise ValueError("text workload must use one byte-identical prompt")
    if len(session_ids) != expected_rows:
        raise ValueError("text workload session IDs must be unique")

    prompt = next(iter(prompts))
    actual_isl = rendered_token_count(tokenizer, prompt)
    if actual_isl != target_isl:
        raise ValueError(f"expected rendered ISL {target_isl}, found {actual_isl}")
    if sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("text workload hash does not match its manifest")

    audit = {
        "model": manifest["model"],
        "request_count": expected_rows,
        "target_isl": target_isl,
        "dataset_sha256": manifest["dataset_sha256"],
    }
    print(
        "WORKLOAD_AUDIT=PASS "
        f"rows={expected_rows} isl={target_isl} sha256={audit['dataset_sha256']}"
    )
    return audit


def generate_workload(
    output_dir: Path,
    model: str = MODEL,
    request_count: int = 1000,
    target_isl: int = 740,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = calibrate_prompt(tokenizer, target_isl)
    dataset_path = output_dir / f"text_{request_count}_isl{target_isl}.jsonl"
    with dataset_path.open("w", encoding="utf-8") as dataset:
        for request_index in range(request_count):
            row = {
                "session_id": f"text-request-{request_index:04d}",
                "text": prompt,
            }
            dataset.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "model": model,
        "request_count": request_count,
        "target_isl": target_isl,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "media_fields": [],
    }
    manifest_path = output_dir / "workload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_workload(dataset_path, manifest_path, tokenizer)
    return dataset_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--request-count", type=int, default=1000)
    parser.add_argument("--target-isl", type=int, default=740)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.validate:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        validate_workload(
            output_dir / f"text_{args.request_count}_isl{args.target_isl}.jsonl",
            output_dir / "workload_manifest.json",
            tokenizer,
        )
        return
    generate_workload(
        output_dir=output_dir,
        model=args.model,
        request_count=args.request_count,
        target_isl=args.target_isl,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize the three pinned Hugging Face datasets for generation and scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import datasets
from datasets import Dataset, load_dataset

from pro_adapter import adapt_row

DATASETS = {
    "verified": {
        "repo": "SWE-bench/SWE-bench_Verified",
        "revision": "91aa3ed51b709be6457e12d00300a6a596d4c6a3",
        "expected": 500,
    },
    "multilingual": {
        "repo": "SWE-bench/SWE-bench_Multilingual",
        "revision": "2b7aced941b4873e9cad3e76abbae93f481d1beb",
        "expected": 300,
    },
    "pro": {
        "repo": "ScaleAI/SWE-bench_Pro",
        "revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
        "expected": 731,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(dataset: Dataset, name: str, expected: int) -> None:
    if len(dataset) != expected:
        raise ValueError(f"{name}: expected {expected} rows, got {len(dataset)}")
    instance_ids = dataset["instance_id"]
    if len(set(instance_ids)) != expected:
        raise ValueError(f"{name}: instance_id values are not unique")


def materialize(dataset: Dataset, name: str, output_root: Path) -> dict:
    agent_dir = output_root / "agent" / name
    evaluator_path = output_root / "evaluator" / f"{name}.jsonl"
    staging = output_root / ".staging" / name

    shutil.rmtree(staging, ignore_errors=True)
    (staging / "data").mkdir(parents=True)
    staged_parquet = staging / "data" / "test-00000-of-00001.parquet"
    staged_jsonl = staging / f"{name}.jsonl"
    dataset.to_parquet(staged_parquet)
    dataset.to_json(staged_jsonl)

    shutil.rmtree(agent_dir, ignore_errors=True)
    agent_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(agent_dir)
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    staged_jsonl = agent_dir / f"{name}.jsonl"
    shutil.move(staged_jsonl, evaluator_path)

    # Exercise the exact local path consumed by mini-SWE-agent.
    reloaded = load_dataset(str(agent_dir), split="test")
    if len(reloaded) != len(dataset):
        raise ValueError(f"{name}: local reload changed row count")

    parquet_path = agent_dir / "data" / "test-00000-of-00001.parquet"
    return {
        "agent_dataset": str(agent_dir),
        "evaluator_dataset": str(evaluator_path),
        "parquet_sha256": sha256(parquet_path),
        "jsonl_sha256": sha256(evaluator_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    provenance = {
        "python": platform.python_version(),
        "datasets_version": datasets.__version__,
        "split": "test",
        "datasets": {},
    }
    for name, spec in DATASETS.items():
        dataset = load_dataset(spec["repo"], revision=spec["revision"], split="test")
        validate(dataset, name, spec["expected"])
        if name == "pro":
            dataset = dataset.map(adapt_row, desc="Adapting SWE-bench Pro")
            for row in dataset:
                if not row["image_name"].startswith("docker.io/jefzda/sweap-images:"):
                    raise ValueError(f"invalid Pro image for {row['instance_id']}")
        outputs = materialize(dataset, name, output_root)
        provenance["datasets"][name] = {**spec, "rows": len(dataset), **outputs}

    shutil.rmtree(output_root / ".staging", ignore_errors=True)
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

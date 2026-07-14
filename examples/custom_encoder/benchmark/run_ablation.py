# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run custom-encoder graph, batching, and bucket-ladder ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.multimodal.sweep.config import BenchmarkConfig  # noqa: E402
from benchmarks.multimodal.sweep.orchestrator import run_sweep  # noqa: E402
from examples.custom_encoder.benchmark.run_image_sweep import (  # noqa: E402
    MODEL,
    RATES,
    REPO_ROOT,
    _config,
    _metadata,
)

Variant = tuple[str, str, int, bool]

VARIANTS: Final[tuple[Variant, ...]] = (
    ("custom-eager-b1", "1", 1, True),
    ("custom-eager-b8", "1,2,4,8", 8, True),
    ("custom-graph-b1", "1", 1, False),
    ("custom-graph-b8-only", "8", 8, False),
    ("custom-graph-1-8", "1,8", 8, False),
    ("custom-graph-full", "1,2,4,8", 8, False),
)


def run_ablation(
    workload_dir: Path,
    output_dir: Path,
    model: str,
    rates: tuple[int, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(model, rates, False, workload_dir)
    metadata["variants"] = [
        {
            "label": label,
            "graph_buckets": buckets,
            "max_batch_cost": max_batch_cost,
            "cuda_graphs_disabled": disabled,
        }
        for label, buckets, max_batch_cost, disabled in VARIANTS
    ]
    metadata_path = output_dir / "ablation_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                "refusing to resume ablation with different provenance: "
                f"existing={existing}, requested={metadata}"
            )
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    workflow = REPO_ROOT / "examples/custom_encoder/launch/agg_qwen2_vl.sh"
    for rate in rates:
        input_file = workload_dir / f"image_custom_qps{rate}_1000_isl515.jsonl"
        for label, buckets, max_batch_cost, disabled in VARIANTS:
            config = _config(
                model,
                input_file,
                rate,
                output_dir,
                [BenchmarkConfig(label=label, workflow=str(workflow))],
                False,
            )
            config.env.update(
                {
                    "DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS": buckets,
                    "DYN_QWEN2_VL_MAX_BATCH_COST": str(max_batch_cost),
                    "DYN_QWEN2_VL_DISABLE_CUDA_GRAPHS": "1" if disabled else "0",
                }
            )
            config.validate(repo_root=REPO_ROOT)
            run_sweep(config, repo_root=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    args = parser.parse_args()
    run_ablation(
        workload_dir=args.workload_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        model=args.model,
        rates=tuple(args.rates),
    )


if __name__ == "__main__":
    main()

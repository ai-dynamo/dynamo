# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate every custom-encoder ablation AIPerf artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_encoder.benchmark.run_ablation import VARIANTS  # noqa: E402
from examples.custom_encoder.benchmark.run_image_sweep import RATES  # noqa: E402
from examples.custom_encoder.benchmark.validate_results import (  # noqa: E402
    validate_result,
)


def validate_ablation(root: Path) -> list[dict[str, object]]:
    labels = tuple(variant[0] for variant in VARIANTS)
    results = [
        validate_result(
            path,
            expected_runtimes=labels,
            expected_rates=RATES,
        )
        for path in sorted(root.rglob("profile_export_aiperf.json"))
        if path.parents[1].name in labels
    ]
    expected = {(label, rate) for label in labels for rate in RATES}
    observed = {(str(result["runtime"]), int(result["rate"])) for result in results}
    rejected = [result for result in results if not result["accepted"]]
    if observed != expected or len(results) != len(expected) or rejected:
        raise AssertionError(
            f"invalid ablation matrix: cells={len(results)} "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)} "
            f"rejected={rejected}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = validate_ablation(root)
    output = root / "ablation_validation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"ABLATION_AUDIT=PASS cells={len(results)} validation={output}")


if __name__ == "__main__":
    main()

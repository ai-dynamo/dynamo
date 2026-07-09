# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import sample_directory
from .campaign import build_sweep_schedule, campaign_summary, run_sample
from .protocol import MODE_ALIASES

async def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    connections = args.connections_sweep or (args.connections,)
    pipelines = args.pipeline_sweep or (args.pipeline,)
    schedule = build_sweep_schedule(connections, pipelines, args.repetitions)
    with tempfile.TemporaryDirectory(prefix="dynkv-saturation-") as temporary:
        campaign_root = (
            Path(
                tempfile.mkdtemp(
                    prefix="dynkv-saturation-campaign-", dir=args.data_root
                )
            )
            if args.data_root is not None
            else Path(temporary)
        )
        samples = []
        for sample_index, (repetition, connection, pipeline) in enumerate(
            schedule, start=1
        ):
            directory = sample_directory(None, campaign_root, sample_index)
            try:
                sample = await run_sample(
                    args,
                    connections=connection,
                    pipeline=pipeline,
                    repetition=repetition,
                    directory=directory,
                )
            except Exception as error:
                sample = {
                    "schema_version": 2,
                    "status": "invalid",
                    "repetition": repetition,
                    "mode": MODE_ALIASES.get(args.mode, args.mode),
                    "connections": connection,
                    "pipeline": pipeline,
                    "error": f"{type(error).__name__}: {error}",
                }
            samples.append(sample)
    if len(samples) == 1:
        return samples[0]
    valid_samples = [sample for sample in samples if sample.get("status") == "ok"]
    invalid_samples = [
        index
        for index, sample in enumerate(samples, start=1)
        if sample.get("status") != "ok"
    ]
    return {
        "schema_version": 2,
        "status": "ok" if not invalid_samples else "invalid",
        "mode": MODE_ALIASES.get(args.mode, args.mode),
        "repetitions": args.repetitions,
        "schedule": [
            {
                "repetition": repetition,
                "connections": connection,
                "pipeline": pipeline,
            }
            for repetition, connection, pipeline in schedule
        ],
        "samples": samples,
        "invalid_samples": invalid_samples,
        "summary": campaign_summary(valid_samples) if valid_samples else None,
    }

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend marker used only after the source-mounted power gate execs it."""

from __future__ import annotations

import argparse
import ctypes
import json
from datetime import datetime, timezone
from pathlib import Path

import pynvml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-max-watts", type=int, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    pynvml.nvmlInit()
    try:
        caps = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            caps.append(round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000))
        if not caps or any(cap > args.expected_max_watts for cap in caps):
            raise RuntimeError(
                f"backend observed caps {caps}, expected <= {args.expected_max_watts}W"
            )

        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda_result = int(cuda.cuInit(0))
        if cuda_result != 0:
            raise RuntimeError(f"cuInit returned {cuda_result}")

        args.marker.write_text(
            json.dumps(
                {
                    "startedAt": datetime.now(timezone.utc).isoformat(),
                    "observedCapsWatts": caps,
                    "cuInitReturn": cuda_result,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    raise SystemExit(main())

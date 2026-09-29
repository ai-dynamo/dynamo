# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic import/arithmetic probe; caller supplies preload and a 90s timeout."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def report(phase, **details):
    print(json.dumps({"phase": phase, "pid": os.getpid(), **details}), flush=True)


def driver_maps():
    with open("/proc/self/maps") as mappings:
        paths = sorted(
            {
                fields[5]
                for line in mappings
                if len(fields := line.rstrip().split(maxsplit=5)) == 6
                and ("/libcuda.so" in fields[5] or "/libnvidia-" in fields[5])
            }
        )
    report("driver_maps", libraries=paths)
    drivers = [path for path in paths if "/libcuda.so" in path]
    if len(drivers) != 1 or not drivers[0].endswith("/libcuda.so.595.58.03"):
        raise RuntimeError(f"expected exactly one host595 libcuda, got {drivers}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi", action="store_true", help="also probe one MPI child")
    args = parser.parse_args()
    if args.mpi:
        # Launch before importing CUDA/MPI in this parent, avoiding post-init fork.
        report("mpi_start")
        subprocess.run(
            [
                "/opt/dynamo/mpi/bin/mpiexec",
                "--allow-run-as-root",
                "-n",
                "1",
                sys.executable,
                str(Path(__file__).resolve()),
            ],
            check=True,
            timeout=45,
        )
        report("mpi_pass")
    # Dynamic imports deliberately expose progress before potentially slow imports.
    report("trtllm_import_start")
    trtllm = importlib.import_module("tensorrt_llm")
    report("trtllm_import_pass", version=trtllm.__version__)
    report("torch_import_start")
    torch = importlib.import_module("torch")
    report("torch_import_pass", version=torch.__version__, cuda=torch.version.cuda)
    driver_maps()
    report("gpu_start")
    values = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) % 17
    gpu_values = values.cuda()
    actual = ((gpu_values @ gpu_values.T) * 2 + 3).cpu()
    torch.testing.assert_close(actual, (values @ values.T) * 2 + 3, rtol=0, atol=0)
    torch.cuda.synchronize()
    driver_maps()
    report("gpu_pass", device=torch.cuda.get_device_name(), shape=list(actual.shape))
    report("pass", scope="TRTLLM import and GPU arithmetic only; not Snapshot")


if __name__ == "__main__":
    main()

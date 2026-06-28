# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from pathlib import Path
from typing import Any

from nccl_checkpoint import NCCLCheckpointLibrary
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary


def symbol_dso_path(symbol: Any) -> Path:
    path = NCCLLibrary._get_symbol_dso_path(symbol, ctypes.CDLL(None))
    if not path:
        raise RuntimeError(f"dladdr did not resolve a DSO for {symbol!r}")
    return Path(path).resolve()


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return left == right


def main() -> None:
    shim_path = Path(os.environ["NCCL_CHECKPOINT_SHIM"]).resolve()
    checkpoint_version = NCCLCheckpointLibrary().get_version()
    library = NCCLLibrary()
    nccl_version = library.ncclGetVersion()

    all_reduce_path = symbol_dso_path(library._funcs["ncclAllReduce"])
    bound_version_path = symbol_dso_path(library._funcs["ncclGetVersion"])
    real_version_path = symbol_dso_path(library.real_lib.ncclGetVersion)

    if not same_file(all_reduce_path, shim_path):
        raise RuntimeError(
            f"ncclAllReduce resolved from {all_reduce_path}, expected {shim_path}"
        )
    if not same_file(bound_version_path, real_version_path):
        raise RuntimeError(
            "ncclGetVersion did not bind directly from real NCCL: "
            f"bound={bound_version_path}, real={real_version_path}"
        )
    if same_file(bound_version_path, shim_path):
        raise RuntimeError(
            f"ncclGetVersion unexpectedly resolved from checkpoint shim {shim_path}"
        )

    print(f"NCCLCheckpoint version: {checkpoint_version}")
    print(f"Real NCCL version: {nccl_version} ({bound_version_path})")
    print(f"Checkpoint ncclAllReduce: {all_reduce_path}")


if __name__ == "__main__":
    main()

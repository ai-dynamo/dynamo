# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
from typing import Any

from nccl_checkpoint import NCCLCheckpointLibrary
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

EXPECTED_CHECKPOINT_VERSION = 100
FLASHINFER_VERSION_FILE = Path("/opt/dynamo/flashinfer-source-version.txt")
EXPECTED_SHIM_PATH = Path(
    "/opt/nccl-checkpoint/lib/libnccl-checkpoint-shim.so"
).resolve()


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


def nccl_version_code(version: str) -> int:
    major, minor, patch = (int(part) for part in version.split("."))
    if major <= 2 and minor <= 8:
        return major * 1000 + minor * 100 + patch
    return major * 10000 + minor * 100 + patch


def require_distribution(name: str, expected_version: str) -> metadata.Distribution:
    canonical_name = name.lower().replace("_", "-").replace(".", "-")
    distributions = [
        distribution
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
        .lower()
        .replace("_", "-")
        .replace(".", "-")
        == canonical_name
    ]
    if len(distributions) != 1:
        raise RuntimeError(
            f"Expected one {name} distribution, found {len(distributions)}"
        )
    distribution = distributions[0]
    if distribution.version != expected_version:
        raise RuntimeError(
            f"{name} version is {distribution.version}, expected {expected_version}"
        )
    print(
        f"{name}: {distribution.version} "
        f"({distribution.locate_file('').resolve()})"
    )
    return distribution


def check_flashinfer() -> None:
    if not FLASHINFER_VERSION_FILE.is_file():
        raise RuntimeError(
            f"Missing custom FlashInfer version file {FLASHINFER_VERSION_FILE}"
        )
    expected_version = FLASHINFER_VERSION_FILE.read_text().strip()
    if not expected_version:
        raise RuntimeError(f"Empty FlashInfer version file {FLASHINFER_VERSION_FILE}")

    require_distribution("flashinfer-python", expected_version)
    require_distribution("flashinfer-cubin", expected_version)

    stale = [
        (distribution.metadata["Name"], distribution.version)
        for distribution in metadata.distributions()
        if distribution.metadata["Name"].lower().startswith("flashinfer")
        and distribution.version.lower().startswith("0.6.12")
    ]
    if stale:
        raise RuntimeError(f"Stale FlashInfer 0.6.12 distributions remain: {stale}")

    for module_name in ("flashinfer", "flashinfer_cubin"):
        module = importlib.import_module(module_name)
        module_version = getattr(module, "__version__", None)
        module_path = Path(module.__file__).resolve()
        print(f"{module_name} import: {module_version} ({module_path})")
        if module_version != expected_version:
            raise RuntimeError(
                f"{module_name} import version is {module_version}, "
                f"expected {expected_version}"
            )


def check_torch(expected_nccl_code: int) -> None:
    expected_local_version = os.environ["EXPECTED_TORCH_LOCAL_VERSION"]
    if not expected_local_version:
        raise RuntimeError(
            "EXPECTED_TORCH_LOCAL_VERSION must be set for the image check"
        )
    torch_distribution = metadata.distribution("torch")
    torch_local_version = torch_distribution.version.partition("+")[2].lower()
    if torch_local_version != expected_local_version:
        raise RuntimeError(
            f"torch version is {torch_distribution.version}, expected local "
            f"version +{expected_local_version}"
        )

    import torch

    torch_nccl_version = torch.cuda.nccl.version()
    if not isinstance(torch_nccl_version, tuple) or len(torch_nccl_version) < 3:
        raise RuntimeError(
            f"Unexpected torch.cuda.nccl.version(): {torch_nccl_version!r}"
        )
    torch_nccl_code = nccl_version_code(
        ".".join(str(part) for part in torch_nccl_version[:3])
    )
    print(
        f"torch: {torch.__version__} ({Path(torch.__file__).resolve()}); "
        f"torch.cuda.nccl.version(): {torch_nccl_version}"
    )
    if torch_nccl_code != expected_nccl_code:
        raise RuntimeError(
            "torch.cuda.nccl.version() does not match installed NCCL: "
            f"torch={torch_nccl_version} ({torch_nccl_code}), "
            f"installed={expected_nccl_code}"
        )


def main() -> None:
    shim_path = Path(os.environ["NCCL_CHECKPOINT_SHIM"]).resolve()
    if shim_path != EXPECTED_SHIM_PATH:
        raise RuntimeError(
            f"NCCL checkpoint shim is {shim_path}, expected {EXPECTED_SHIM_PATH}"
        )

    expected_nccl_version = os.environ.get("EXPECTED_NCCL_VERSION")
    if not expected_nccl_version:
        raise RuntimeError(
            "EXPECTED_NCCL_VERSION must be set for the PyNCCL binding check"
        )
    expected_nccl_code = nccl_version_code(expected_nccl_version)
    require_distribution("nvidia-nccl-cu13", expected_nccl_version)

    checkpoint_version = NCCLCheckpointLibrary().get_version()
    if (
        checkpoint_version.checkpoint_version != EXPECTED_CHECKPOINT_VERSION
        or checkpoint_version.nccl_version != expected_nccl_code
    ):
        raise RuntimeError(
            f"Unexpected NCCLCheckpoint version: {checkpoint_version}; expected "
            f"checkpoint={EXPECTED_CHECKPOINT_VERSION}, NCCL={expected_nccl_code}"
        )

    library = NCCLLibrary()
    nccl_raw_version = library.ncclGetRawVersion()
    nccl_version = library.ncclGetVersion()
    if nccl_raw_version != expected_nccl_code or nccl_version != expected_nccl_version:
        raise RuntimeError(
            f"PyNCCL reports NCCL {nccl_version} ({nccl_raw_version}), expected "
            f"{expected_nccl_version} ({expected_nccl_code})"
        )

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
    print(
        f"Real NCCL version: {nccl_version} ({nccl_raw_version}) "
        f"({bound_version_path})"
    )
    print(f"Checkpoint ncclAllReduce: {all_reduce_path}")
    check_flashinfer()
    check_torch(expected_nccl_code)


if __name__ == "__main__":
    main()

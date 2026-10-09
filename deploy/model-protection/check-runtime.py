# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify protected overlay capabilities without loading a model or GPU."""

import os
from importlib.metadata import version

import dynamo._core as core
from dynamo.vllm import protection_bootstrap


def main() -> None:
    expected = os.environ["EXPECTED_VLLM_VERSION"]
    actual = version("vllm")
    tpm_enabled = getattr(core, "model_protection_tpm_enabled", False)
    runtime_enabled = getattr(core, "model_protection_runtime_enabled", False)
    loader_version = getattr(protection_bootstrap, "SUPPORTED_VLLM_VERSION", None)
    print(
        f"protected runtime: vLLM={actual}, expected={expected}, "
        f"loader={loader_version}, TPM={tpm_enabled}, layers={runtime_enabled}, "
        f"extension={core.__file__}",
        flush=True,
    )
    if actual != expected:
        raise SystemExit(
            f"vLLM version mismatch: installed={actual}, expected={expected}"
        )
    if not tpm_enabled:
        raise SystemExit(
            "TPM feature missing from imported dynamo._core; check the wheel build features and Python import path"
        )
    if not runtime_enabled:
        raise SystemExit(
            "Layer runtime missing from dynamo._core; rebuild the runtime wheel from this source"
        )
    if loader_version != expected:
        raise SystemExit(
            "Protected loader version mismatch; rebuild the base with SKIP_BASE_BUILD=false"
        )


if __name__ == "__main__":
    main()

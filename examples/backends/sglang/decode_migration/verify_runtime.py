# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

SOURCE_ROOT = Path("/opt/sglang/python")
COMPILED_DEPENDENCIES = {
    "flashinfer-cubin",
    "flashinfer-python",
    "sgl-deep-gemm",
    "sglang-kernel",
}


def source_requirements() -> dict[str, Requirement]:
    pyproject = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text())
    requirements = map(Requirement, pyproject["project"]["dependencies"])
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) in COMPILED_DEPENDENCIES
    }


def main() -> None:
    requirements = source_requirements()
    if requirements.keys() != COMPILED_DEPENDENCIES:
        missing = sorted(COMPILED_DEPENDENCIES - requirements.keys())
        raise RuntimeError(f"SGLang source is missing dependency pins: {missing}")

    installed = {name: version(name) for name in sorted(requirements)}
    for name, requirement in requirements.items():
        if installed[name] not in requirement.specifier:
            raise RuntimeError(
                f"{name}=={installed[name]} does not satisfy {requirement.specifier}"
            )

    deep_gemm = importlib.import_module("deep_gemm")
    if not callable(getattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked", None)):
        raise RuntimeError("DeepGEMM masked BF16 grouped GEMM is unavailable")

    modules = {
        name: importlib.import_module(name).__file__
        for name in (
            "dynamo._core",
            "dynamo.sglang.request_handlers.llm.decode_handler",
            "nixl",
            "sglang.srt.disaggregation.decode_migration",
        )
    }
    print(json.dumps({"packages": installed, "modules": modules}, indent=2))


if __name__ == "__main__":
    main()

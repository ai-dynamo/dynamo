#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fail fast if the UE8M0 flag is wired to the wrong cache macro."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def macro_body(source: str, name: str) -> str:
    match = re.search(rf"#define {name}\(.*?(?=\n\n)", source, re.DOTALL)
    assert match is not None, f"missing macro {name}"
    return match.group(0)


def main() -> None:
    path = Path(sys.argv[1])
    source = path.read_text()
    ordinary = macro_body(source, "CALL_CONCAT_AND_CACHE_MLA")
    ds_mla = macro_body(source, "CALL_CONCAT_AND_CACHE_DS_MLA")

    assert "use_ue8m0" not in ordinary, (
        "UE8M0 flag must not be passed to concat_and_cache_mla_kernel"
    )
    assert ds_mla.count("use_ue8m0") == 1, (
        "UE8M0 flag must be passed exactly once to concat_and_cache_ds_mla_kernel"
    )
    signature = re.search(
        r"__global__ void concat_and_cache_ds_mla_kernel\((.*?)\n\)",
        source,
        re.DOTALL,
    )
    assert signature is not None and "const bool use_ue8m0" in signature.group(1)
    assert re.search(
        r'if \(kv_cache_dtype == "fp8_ds_mla"\).*?'
        r"const bool use_ue8m0 = ds_mla_use_ue8m0_scale\(\);.*?"
        r"CALL_CONCAT_AND_CACHE_DS_MLA",
        source,
        re.DOTALL,
    ), "fp8_ds_mla dispatch does not define and use the UE8M0 flag"
    print("FP8_DS_MLA_UE8M0_SOURCE_WIRING_OK")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the Dynamo SGLang sidecar from Python."""

import sys

from dynamo._core import run_sglang_sidecar
from dynamo.runtime.logging import configure_dynamo_logging

def main(argv: list[str] | None = None) -> None:
    configure_dynamo_logging(service_name="dynamo.sglang_sidecar")
    run_sglang_sidecar(sys.argv[1:] if argv is None else argv)

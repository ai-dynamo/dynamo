# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Public Python entrypoint for the native SGLang backend."""

from collections.abc import Sequence


def run(args: Sequence[str]) -> None:
    from dynamo._core import run_sglang_backend
    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging(service_name="dynamo.native_sglang_backend")
    run_sglang_backend(list(args))

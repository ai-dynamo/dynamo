# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections.abc import Sequence

os.environ.setdefault("DYNAMO_SKIP_PYTHON_LOG_INIT", "1")

from dynamo.llm import run_kv_indexer


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        run_kv_indexer(args)
    except Exception as exc:
        if "-h" in args or "--help" in args:
            print(exc)
            return 0
        raise
    return 0

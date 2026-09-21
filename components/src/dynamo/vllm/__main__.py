# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    import sys

    from dynamo.vllm.protection_bootstrap import prepare_model_argv

    protection = prepare_model_argv(sys.argv[1:])
    try:
        from dynamo.common.snapshot.restore_context import (
            maybe_run_restore_standby_mode,
        )

        # Check before importing dynamo.vllm.main: restore standby mode must capture
        # env and hold without importing vLLM or constructing backend/runtime state.
        maybe_run_restore_standby_mode()

        from dynamo.vllm.main import main

        main(protection)
    except BaseException:
        try:
            protection.cleanup()
        except Exception:
            sys.stderr.write("model_protection event=cleanup_failed\n")
        raise

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the TensorRT-LLM backend.

Usage:
    python -m dynamo.trtllm.unified_main <trtllm args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""


def main():
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    maybe_run_restore_standby_mode()

    from dynamo.common.backend.run import run
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    run(TrtllmLLMEngine)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit entrypoint guidance for engine-specific OpenAI backend launchers."""

from collections.abc import Sequence

MESSAGE = (
    "Use 'python -m dynamo.openai_backend.sglang' or "
    "'python -m dynamo.openai_backend.vllm'. "
    "The generic 'dynamo.openai_backend' entrypoint is not engine-specific."
)


def main(argv: Sequence[str] | None = None) -> None:
    del argv
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()

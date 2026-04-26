# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Public CLI entrypoint for the OpenAI-compatible backend launcher."""

from collections.abc import Sequence

from dynamo.openai_backend.launcher import launch_main


def main(argv: Sequence[str] | None = None) -> None:
    launch_main(argv)


if __name__ == "__main__":
    main()

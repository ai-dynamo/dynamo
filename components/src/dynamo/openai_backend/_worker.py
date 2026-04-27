# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Private subprocess entrypoint for the OpenAI-compatible backend worker."""

from dynamo.openai_backend.worker import worker_main


if __name__ == "__main__":
    worker_main()

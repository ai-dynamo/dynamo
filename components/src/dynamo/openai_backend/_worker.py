# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private subprocess entrypoint for the OpenAI-compatible backend worker."""

from dynamo.openai_backend.worker import worker_main

if __name__ == "__main__":
    worker_main()

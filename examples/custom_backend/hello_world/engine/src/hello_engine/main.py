# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.common.backend.run import run

from .engine import HelloEngine


def main() -> None:
    run(HelloEngine)


if __name__ == "__main__":
    main()

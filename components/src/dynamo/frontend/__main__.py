#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dynamo._jemalloc import maybe_preload_jemalloc

if __name__ == "__main__":
    # DYN_FRONTEND_JEMALLOC predates DYN_JEMALLOC and remains a frontend-only alias.
    maybe_preload_jemalloc(alias="DYN_FRONTEND_JEMALLOC")

    # Import the runtime only after configuring the allocator.
    from dynamo.frontend.main import main

    main()

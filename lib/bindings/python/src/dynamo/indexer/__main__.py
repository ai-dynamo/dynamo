# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo._jemalloc import maybe_preload_jemalloc

if __name__ == "__main__":
    maybe_preload_jemalloc()

    # Import the runtime only after configuring the allocator.
    from dynamo.indexer.main import main

    raise SystemExit(main())

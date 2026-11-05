#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dynamo._jemalloc import maybe_preload

if __name__ == "__main__":
    # Re-execs with the bundled jemalloc, so it must run before the runtime is imported.
    maybe_preload()

    from dynamo.frontend.main import main

    main()

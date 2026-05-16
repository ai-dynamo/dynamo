#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Activate jemalloc as the process-wide allocator (issue #9466). Run before
# any imports to save unnecessary startup overhead if re-execution is needed.
# See dynamo/common/allocator.py for details.
from dynamo.common.allocator import maybe_reexec_with_jemalloc

maybe_reexec_with_jemalloc("dynamo.frontend")

from dynamo.frontend.main import main  # noqa: E402

if __name__ == "__main__":
    main()

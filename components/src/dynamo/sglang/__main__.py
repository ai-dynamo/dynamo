#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if os.environ.get("DYN_READY_FOR_CHECKPOINT_FILE"):
    from dynamo.sglang.patches import apply_snapshot_patches

    apply_snapshot_patches()

from dynamo.sglang.main import main

if __name__ == "__main__":
    main()

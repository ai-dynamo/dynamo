#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

from dynamo.common.utils.snapshot.restore_context import (
    is_restore_placeholder_mode,
    run_restore_placeholder,
)

# Check before importing dynamo.sglang.main: the restore placeholder must
# capture env and hold without importing SGLang or constructing backend state.
if is_restore_placeholder_mode():
    run_restore_placeholder()

from dynamo.sglang.main import main

if __name__ == "__main__":
    main()

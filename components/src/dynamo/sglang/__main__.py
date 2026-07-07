#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # Check before importing dynamo.sglang.main: restore standby mode must
    # capture env and hold without importing SGLang or constructing backend state.
    maybe_run_restore_standby_mode()

    from dynamo.sglang.main import main

    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if (
    "NCCL_RUNTIME_CONNECT" not in os.environ
    and os.environ.get("GMS_SOCKET_DIR")
    and os.environ.get("ENGINE_ID") is not None
    and os.environ.get("FAILOVER_LOCK_PATH")
):
    # GMS shadow engines park before serving, so lazy NCCL connection setup
    # keeps the parked duplicate CUDA/NCCL footprint smaller.
    os.environ["NCCL_RUNTIME_CONNECT"] = "1"

from dynamo.trtllm.main import main

if __name__ == "__main__":
    main()

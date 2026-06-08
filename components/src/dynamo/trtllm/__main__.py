# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

from dynamo.trtllm.startup import configure_gms_openmpi_defaults

configure_gms_openmpi_defaults()

from dynamo.trtllm.main import main

if __name__ == "__main__":
    main()

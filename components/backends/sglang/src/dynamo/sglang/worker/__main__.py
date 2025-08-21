#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


import logging

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.main import main

if __name__ == "__main__":
    configure_dynamo_logging()

    logging.warning(
        "DEPRECATION WARNING: `python3 -m dynamo.sglang.worker` is deprecated and will be removed in dynamo v0.5.0."
        "Use `python3 -m dynamo.sglang` instead.",
    )
    main()

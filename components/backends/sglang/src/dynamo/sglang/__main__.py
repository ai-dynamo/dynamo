#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import asyncio

import uvloop

from dynamo.sglang.components.worker import worker

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())

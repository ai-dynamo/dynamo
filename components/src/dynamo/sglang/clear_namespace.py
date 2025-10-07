#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


@dynamo_worker()
async def clear_namespace(runtime: DistributedRuntime, namespace: str):
    await runtime.temp_clear_namespace(f"/{namespace}/")
    logging.info(f"Cleared /{namespace}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, required=True)
    args = parser.parse_args()
    assert (
        args.namespace
    ), "Missing namespace, either pass --namespace or set DYN_NAMESPACE"
    asyncio.run(clear_namespace(args.namespace))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import uvloop
from protocols import TransferConfig

from dynamo.runtime import DistributedRuntime, dynamo_worker

NUM_REQUESTS = 100


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # Get endpoint (sender -> receiver)
    sender_endpoint = runtime.endpoint("embedding_transfer.sender.generate")
    receiver_endpoint = runtime.endpoint("embedding_transfer.receiver.generate")
    sender_update_config_endpoint = runtime.endpoint(
        "embedding_transfer.sender.update_config"
    )
    receiver_update_config_endpoint = runtime.endpoint(
        "embedding_transfer.receiver.update_config"
    )

    # Create client and wait for service to be ready
    sender_client = await sender_endpoint.client()
    await sender_client.wait_for_instances()
    receiver_client = await receiver_endpoint.client()
    await receiver_client.wait_for_instances()
    sender_update_config_client = await sender_update_config_endpoint.client()
    await sender_update_config_client.wait_for_instances()
    receiver_update_config_client = await receiver_update_config_endpoint.client()
    await receiver_update_config_client.wait_for_instances()

    # NOTE From CPU is not the same as E/PD, E/PD originates from GPU and has
    # GPU to CPU copy
    for transfer_type in ["local", "nixl_write", "nixl_read"]:
        for workflow_string, client in [
            ("receiver-first", receiver_client),
            ("sender-first", sender_client),
        ]:
            for use_gpu in [False, True]:
                # Update sender/receiver config before each run
                config = TransferConfig(
                    use_gpu=use_gpu,
                    tensor_count_per_request=30,
                    transfer_type=transfer_type,
                )
                async for res in await sender_update_config_client.round_robin(
                    config.model_dump_json()
                ):
                    pass
                async for res in await receiver_update_config_client.round_robin(
                    config.model_dump_json()
                ):
                    pass

                if transfer_type == "nixl_read" and use_gpu:
                    print(
                        f"Skipping: use_gpu={use_gpu} with transfer type: {transfer_type}"
                    )
                    print(
                        "Reason: nixl_connect errors out on GPU tensor, i.e. NIXL_ERR_NOT_ALLOWED"
                    )
                    continue

                num_requests = NUM_REQUESTS
                try:
                    print(
                        f"Workflow: {workflow_string}, From GPU: {use_gpu}, Transfer Type: {transfer_type}"
                    )
                    # warm up
                    async for response in await client.round_robin(
                        "world,sun,moon,star"
                    ):
                        continue
                    start_time = time.perf_counter()
                    streams = [
                        await client.round_robin("world,sun,moon,star")
                        for _ in range(num_requests)
                    ]
                    for stream in streams:
                        async for response in stream:
                            continue
                    end_time = time.perf_counter()
                    print(f"Time taken: {end_time - start_time:.2f} seconds")
                except Exception as e:
                    # Log the exception with context
                    print(f"Error in worker: {type(e).__name__}: {e}")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())

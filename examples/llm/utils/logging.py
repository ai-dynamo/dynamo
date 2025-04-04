import asyncio
from dynamo._core import Client


async def check_required_workers(
    workers_client: Client, required_workers: int, on_change=True, poll_interval=0.5
):
    """Wait until the minimum number of workers are ready."""
    worker_ids = workers_client.endpoint_ids()
    num_workers = len(worker_ids)

    while num_workers < required_workers:
        await asyncio.sleep(poll_interval)
        worker_ids = workers_client.endpoint_ids()
        new_count = len(worker_ids)

        if (not on_change) or new_count != num_workers:
            print(
                f"Waiting for more workers to be ready.\n"
                f" Current: {new_count},"
                f" Required: {required_workers}"
            )
        num_workers = new_count

    return worker_ids

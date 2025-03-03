import asyncio

import uvloop
from processor_backend.common import parse_vllm_args

from triton_distributed.runtime import (
    DistributedRuntime,
    ModelDeploymentCard,
    OAIChatPreprocessor,
    triton_worker,
)

uvloop.install()


@triton_worker()
async def preprocessor(runtime: DistributedRuntime, model_name: str, model_path: str):
    # create model deployment card
    mdc = await ModelDeploymentCard.from_local_path(model_path, model_name)
    # create preprocessor endpoint
    component = runtime.namespace("triton-init").component("preprocessor")
    await component.create_service()
    endpoint = component.endpoint("generate")

    # create backend endpoint
    backend = runtime.namespace("triton-init").component("backend").endpoint("generate")

    # start preprocessor service with next backend
    chat = OAIChatPreprocessor(mdc, endpoint, next=backend)
    await chat.start()


if __name__ == "__main__":
    args = parse_vllm_args()
    asyncio.run(preprocessor(args.model, args.model_path))

from typing import AsyncGenerator

from dynamo.sdk import depends, dynamo_endpoint, service
from examples.hello_world_components.components.abstract_component import GeneratorService, RequestType, ResponseType


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
)
class Middle2: #(GeneratorService): TODO hutm: make it a subclass of GeneratorService
    backend = depends(GeneratorService)

    def __init__(self) -> None:
        print("Starting middle2")
        #TODO hutm: has to be applied in the decorator
        self.backend = Middle2.dependencies["backend"].get()

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        req_text = req.text
        print(f"Middle2 received: {req_text}")
        text = f"{req_text}-mid2"
        next_request = RequestType(text=text).model_dump_json()
        async for response in self.backend.generate(next_request):
            print(f"Middle2 received response: {response}")
            yield f"Middle2: {response}"

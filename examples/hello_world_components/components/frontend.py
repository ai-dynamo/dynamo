from typing import AsyncGenerator

from dynamo.sdk import api, depends, dynamo_endpoint, service
from examples.hello_world_components.components.abstract_component import GeneratorService, RequestType


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 60},
)
class Frontend:
    middle = depends(GeneratorService)

    def __init__(self) -> None:
        print("Starting frontend")
        #TODO: hutm has to be applied in the decorator
        self.middle = Frontend.dependencies["middle"].get()

    # Regular HTTP API
    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"

from typing import AsyncGenerator

from components.abstract_component import GeneratorService
from dynamo.sdk import dynamo_endpoint, service
from examples.hello_world_components.components.abstract_component import RequestType, ResponseType


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=1,
)
class Backend: #(GeneratorService): TODO hutm: make it a subclass of GeneratorService
    def __init__(self) -> None:
        print("Starting backend")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        print(f"Backend received: {req_text}")
        text = f"{req_text}-back"
        for token in text.split():
            yield f"Backend: {token}"

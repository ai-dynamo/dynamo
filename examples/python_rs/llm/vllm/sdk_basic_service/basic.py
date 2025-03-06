from compoundai import api
from compoundai.sdk.decorators import dynemo_endpoint
from compoundai.sdk.dependency import depends
from compoundai.sdk.service import service
from pydantic import BaseModel

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ dynemo/distributed-runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ dynemo/distributed-runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
"""


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynemo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=3,
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @dynemo_endpoint()
    async def generate(self, text: RequestType):
        """Generate tokens."""
        text = f"{text.text}-back"
        print(f"Backend received: {text}")
        for token in text.split():
            yield f"Backend: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynemo={"enabled": True, "namespace": "inference"},
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle")

    @dynemo_endpoint()
    async def generate(self, text: RequestType):
        """Forward requests to backend."""
        text = f"{text.text}-mid"
        print(f"Middle received: {text}")
        txt = RequestType(text=text)
        async for response in self.backend.generate(txt.model_dump_json()):
            print(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(resources={"cpu": "1"}, traffic={"timeout": 60})  # Regular HTTP API
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"

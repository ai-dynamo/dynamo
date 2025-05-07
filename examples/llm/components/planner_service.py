import logging

from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from components.planner import start_planner
from pydantic import BaseModel
from dynamo.sdk.lib.service import ComponentType
logger = logging.getLogger(__name__)

class RequestType(BaseModel):
    text: str

from dynamo.sdk.lib.image import DYNAMO_IMAGE

import argparse
parser = argparse.ArgumentParser()

@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
    component_type=ComponentType.PLANNER
)
class Planner:
    def __init__(self):
        self.runtime = dynamo_context["runtime"]
        self.namespace = "dynamo" # TODO: this needs to be dynamic based on the injected value
        self.args = parser.parse_args([
            "--namespace", self.namespace,          # your chosen namespace
            "--environment", "kubernetes",    # your chosen environment
        ])

    @async_on_start
    async def async_init(self):
        await start_planner(self.runtime, self.args)

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
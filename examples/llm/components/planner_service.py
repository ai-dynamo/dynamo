import logging

from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from components.planner import start_planner
from pydantic import BaseModel
from dynamo.sdk.lib.service import ComponentType
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.runtime.logging import configure_dynamo_logging

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
        "component_type": ComponentType.PLANNER,
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Planner:
    def __init__(self):
        configure_dynamo_logging(service_name="Planner")
        logger.info("Starting planner")
        self.runtime = dynamo_context["runtime"]

        config = ServiceConfig.get_instance()

        # TODO: this should default to whichever namespace this service is actually running in
        # can be passed via CLI arg to dynamo serve with --Planner.namespace and --Planner.environment
        self.namespace = config.get("Planner", {}).get("namespace", "dynamo")
        self.environment = config.get("Planner", {}).get("environment", "local")

        self.args = parser.parse_args([
            "--namespace", self.namespace,          # your chosen namespace
            "--environment", self.environment,    # your chosen environment
        ])

    @async_on_start
    async def async_init(self):
        await start_planner(self.runtime, self.args)

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
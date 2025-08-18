import sglang as sgl
from dataclasses import dataclass

from dynamo.sgl.args import Config


@dataclass
class RequestHandlerConfig:
    """
    Configuration for request handlers
    """

    component: object
    engine: sgl.Engine
    config: Config

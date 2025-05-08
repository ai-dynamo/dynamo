from enum import Enum


class TargetEnum(str, Enum):
    """The target deployment environment for the service"""

    DYNAMO = "dynamo"
    BENTO = "bento"

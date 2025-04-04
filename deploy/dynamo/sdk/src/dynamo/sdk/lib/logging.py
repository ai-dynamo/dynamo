import logging
import logging.config

from dynamo.runtime.logging import configure_logger


# Create a replacement for BentoML's configure_server_logging
def configure_server_logging():
    """
    A single place to configure logging for Dynamo that can be used to replace BentoML's logging configuration.
    """
    # First, remove any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure the logger with Dynamo's handler
    configure_logger()

    # Make sure bentoml's loggers use the same configuration
    bentoml_logger = logging.getLogger("bentoml")
    bentoml_logger.propagate = True  # Make sure logs propagate to the root logger

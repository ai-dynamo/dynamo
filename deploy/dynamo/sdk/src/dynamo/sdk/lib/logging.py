import logging
import logging.config

# Import the Dynamo logger
from dynamo.runtime.logging import LogHandler, configure_logger

# Create a replacement for BentoML's configure_server_logging
def configure_server_logging():
    """
    Replaces BentoML's logging configuration with Dynamo's.
    This ensures all logs go through Dynamo's handler.
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
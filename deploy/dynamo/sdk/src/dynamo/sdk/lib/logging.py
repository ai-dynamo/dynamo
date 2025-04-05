# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.config
import os

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

    # Configure vllm loggers to use our format
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"  # Disable VLLM's default configuration

    # Get all vllm loggers and configure them
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = []
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = True

    # Also handle any root loggers that VLLM might be using without the vllm prefix
    other_loggers = ["__init__", "nixl"]
    for logger_name in other_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = True

    # Bento Logger
    bentoml_logger.setLevel(logging.ERROR)
    bentoml_logger.propagate = True

    # Override internal BentoML loggers
    aux_bento_loggers = ["tag"]
    for logger_name in aux_bento_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = True

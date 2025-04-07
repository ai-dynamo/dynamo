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
    A single place to configure logging for Dynamo.
    """
    # First, remove any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure the logger with Dynamo's handler
    configure_logger()

    # Disable VLLM's default configuration
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    # loggers that should be configured to INFO
    info_lgrs = ["vllm", "nixl", "__init__"]
    for lgr in info_lgrs:
        logger = logging.getLogger(lgr)
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = True

    # loggers that should be configured to ERROR
    error_lgrs = ["bentoml", "tag"]
    for lgr in error_lgrs:
        logger = logging.getLogger(lgr)
        logger.handlers = []
        logger.setLevel(logging.ERROR)
        logger.propagate = True

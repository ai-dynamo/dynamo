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

import json
import logging
import os
import tempfile

from dynamo._core import log_message


class LogHandler(logging.Handler):
    """
    Custom logging handler that sends log messages to the Rust env_logger
    """

    def emit(self, record):
        """
        Emit a log record
        """
        log_entry = self.format(record)
        if record.funcName == "<module>":
            module_path = record.module
        else:
            module_path = f"{record.module}.{record.funcName}"
        log_message(
            record.levelname.lower(),
            log_entry,
            module_path,
            record.pathname,
            record.lineno,
        )


# Configure the Python logger to use the NimLogHandler
def configure_logger(service_name: str | None, worker_id: int | None):
    """
    Called once to configure the Python logger to use the LogHandler
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = LogHandler()

    # Simple formatter without date and level info since it's already provided by Rust
    formatter = logging.Formatter("%(message)s")
    formatter_prefix = construct_formatter_prefix(service_name, worker_id)
    if len(formatter_prefix) != 0:
        formatter = logging.Formatter(f"[{formatter_prefix}] %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)


def construct_formatter_prefix(service_name: str | None, worker_id: int | None) -> str:
    tmp = ""
    if service_name is not None:
        tmp += f"{service_name}"

    if worker_id is not None:
        tmp += f":{worker_id}"

    return tmp.strip()


def configure_dynamo_logging(
    service_name: str | None = None, worker_id: int | None = None
):
    """
    A single place to configure logging for Dynamo.
    """
    # First, remove any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure the logger with Dynamo's handler
    configure_logger(service_name, worker_id)

    # map the DYN_LOG variable to a logging level
    dyn_var = os.environ.get("DYN_LOG", "info")
    dyn_level = log_level_mapping(dyn_var)

    configure_vllm_logging(dyn_level)

    # loggers that should be configured to ERROR
    error_loggers = ["bentoml", "tag"]
    for logger_name in error_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.setLevel(logging.ERROR)
        logger.propagate = True


def log_level_mapping(level: str) -> int:
    """
    The DYN_LOG variable is set using "debug" or "trace" or "info.
    This function maps those to the appropriate logging level and defaults to INFO
    if the variable is not set or a bad value.
    """
    if level == "debug":
        return logging.DEBUG
    elif level == "info":
        return logging.INFO
    elif level == "warn" or level == "warning":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR
    elif level == "critical":
        return logging.CRITICAL
    elif level == "trace":
        return logging.INFO
    else:
        return logging.INFO


def configure_vllm_logging(dyn_level: int):
    """
    vLLM requires a logging config file to be set in the environment.
    This function creates a temporary file with the VLLM logging config and sets the
    VLLM_LOGGING_CONFIG_PATH environment variable to the path of the file.
    """

    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    vllm_level = logging.getLevelName(dyn_level)

    # Create a temporary config file for VLLM
    vllm_config = {
        "formatters": {"simple": {"format": "%(message)s"}},
        "handlers": {
            "dynamo": {
                "class": "dynamo.runtime.logging.LogHandler",
                "formatter": "simple",
                "level": vllm_level,
            }
        },
        "loggers": {
            "vllm": {"handlers": ["dynamo"], "level": vllm_level, "propagate": False}
        },
        "version": 1,
        "disable_existing_loggers": False,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(vllm_config, f)
        os.environ["VLLM_LOGGING_CONFIG_PATH"] = f.name

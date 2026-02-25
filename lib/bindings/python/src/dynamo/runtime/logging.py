# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging.config
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


class VllmColorFormatter(logging.Formatter):
    """Formatter that matches Rust tracing's compact colored output style.

    Used for vLLM logs routed through a StreamHandler (bypassing the Rust
    bridge) so that VLLM_LOGGING_LEVEL is respected independently of DYN_LOG
    while still producing visually consistent colored output.
    """

    # ANSI color codes matching Rust tracing's defaults
    _COLORS = {
        "DEBUG": "\033[2m",  # dim
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
    }
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def format(self, record):
        from datetime import datetime, timezone

        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        level = record.levelname
        color = self._COLORS.get(level, "")
        if record.funcName and record.funcName != "<module>":
            target = f"{record.module}.{record.funcName}"
        else:
            target = record.module
        msg = record.getMessage()
        return (
            f"{self._DIM}{ts}{self._RESET} "
            f"{color}{level:>5}{self._RESET} "
            f"{self._DIM}{target}{self._RESET}{self._DIM}:{self._RESET} "
            f"{msg}"
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

    # configure inference engine loggers
    configure_vllm_logging(dyn_level)
    if not get_bool_env_var("DYN_SKIP_SGLANG_LOG_FORMATTING"):
        configure_sglang_logging(dyn_level)
    if not get_bool_env_var("DYN_SKIP_TRTLLM_LOG_FORMATTING"):
        configure_trtllm_logging(dyn_level)

    # loggers that should be configured to ERROR
    error_loggers = ["tag"]
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


def configure_sglang_logging(dyn_level: int):
    """
    SGLang allows us to create a custom logging config file
    """

    sglang_level = logging.getLevelName(dyn_level)

    sglang_config = {
        "formatters": {"simple": {"format": "%(message)s"}},
        "handlers": {
            "dynamo": {
                "class": "dynamo.runtime.logging.LogHandler",
                "formatter": "simple",
                "level": sglang_level,
            }
        },
        "loggers": {
            "sglang": {
                "handlers": ["dynamo"],
                "level": sglang_level,
                "propagate": False,
            }
        },
        "version": 1,
        "disable_existing_loggers": False,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sglang_config, f)
        os.environ["SGLANG_LOGGING_CONFIG_PATH"] = f.name


def configure_vllm_logging(dyn_level: int):
    """
    Configure vLLM logging for the main process and subprocesses.

    Main process: replaces vLLM's StreamHandler with dynamo's LogHandler so logs
    flow through the Rust logging bridge.

    Subprocesses (EngineCore, workers): use vLLM's DEFAULT_LOGGING_CONFIG
    (StreamHandler to stderr) since the Rust runtime is not initialized there.
    Setting VLLM_CONFIGURE_LOGGING=1 without VLLM_LOGGING_CONFIG_PATH causes
    vLLM to use its built-in default config in spawned subprocesses.

    The dyn_level param is kept for signature compatibility but does not control
    the vLLM logger level. Use VLLM_LOGGING_LEVEL env var instead.
    """
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"

    # vLLM level is controlled exclusively by VLLM_LOGGING_LEVEL.
    # DYN_LOG controls dynamo logging only — it does not affect vLLM.
    vllm_level = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()

    # Route vLLM logs through dynamo's LogHandler (Rust bridge) for consistent
    # colored output.  The Rust env_filter (DYN_LOG) also filters messages, so
    # when VLLM_LOGGING_LEVEL is more verbose than DYN_LOG we must widen the
    # Rust filter for vLLM targets.  Python-side logger level still controls
    # what reaches the handler.
    #
    # The LogHandler sets Rust target to "<module>.<funcName>", so vLLM log
    # records have targets like "loggers.log", "async_llm.check_health", etc.
    # We add a blanket "vllm=" directive isn't possible since targets don't
    # start with "vllm".  Instead, we use a StreamHandler that writes directly
    # to stderr, bypassing the Rust filter, but formatted to match Rust output.
    main_config = {
        "formatters": {
            "vllm": {
                "()": "dynamo.runtime.logging.VllmColorFormatter",
            }
        },
        "handlers": {
            "vllm_stderr": {
                "class": "logging.StreamHandler",
                "formatter": "vllm",
                "stream": "ext://sys.stderr",
            }
        },
        "loggers": {
            "vllm": {
                "handlers": ["vllm_stderr"],
                "level": vllm_level,
                "propagate": False,
            }
        },
        "version": 1,
        "disable_existing_loggers": False,
    }
    logging.config.dictConfig(main_config)

    # Suppress noisy "Called check_health" debug spam from vLLM's AsyncLLM.
    # Dynamo's VllmEngineMonitor calls check_health every 2s which floods
    # the logs at DEBUG level.  vLLM's own server doesn't have this issue
    # because it doesn't run a periodic health check loop.
    class _HealthCheckFilter(logging.Filter):
        def filter(self, record):
            return not (
                record.funcName == "check_health" and record.levelno <= logging.DEBUG
            )

    logging.getLogger("vllm.v1.engine.async_llm").addFilter(_HealthCheckFilter())


def map_dyn_log_to_tllm_level(dyn_log_value: str) -> str:
    """
    Map DYN_LOG string value to TensorRT-LLM log level.

    Args:
        dyn_log_value: The DYN_LOG environment variable value (e.g., "debug", "info,module::path=trace")

    Returns:
        The corresponding TLLM_LOG_LEVEL value (e.g., "VERBOSE", "INFO")
    """
    # Extract the base level (handle cases like "debug,module::path=trace")
    base_level = dyn_log_value.lower().split(",")[0].strip()

    # Map DYN_LOG levels to TLLM_LOG_LEVEL
    level_mapping = {
        "debug": "DEBUG",
        "info": "INFO",
        "warn": "WARNING",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "ERROR",
        "trace": "TRACE",
    }

    return level_mapping.get(base_level, "INFO")


def configure_trtllm_logging(dyn_level: int):
    # Only set TLLM_LOG_LEVEL if it's not already set
    # This allows users to override it explicitly if needed
    if "TLLM_LOG_LEVEL" not in os.environ:
        dyn_level_name = logging.getLevelName(dyn_level)
        tllm_level = map_dyn_log_to_tllm_level(dyn_level_name)
        os.environ["TLLM_LOG_LEVEL"] = tllm_level
        logging.debug(f"Set TLLM_LOG_LEVEL to {tllm_level} based on DYN_LOG")


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        logging.warning(
            f"The environment variable {name} has an unrecognized value={value}, treating as false"
        )

    return value in truthy_values

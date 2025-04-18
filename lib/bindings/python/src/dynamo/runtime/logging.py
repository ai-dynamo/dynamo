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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


DEFAULT_LOG_LEVEL = logging.DEBUG


class Timer:
    """Simple timer implementation that can time interval"""

    def __init__(self, interval_func=None, stop_func=None):
        self.start_time = time.perf_counter()
        self.last_interval_time = self.start_time
        self.interval_func = interval_func
        self.stop_func = stop_func

    def time_interval(self):
        now = time.perf_counter()
        interval_time = now - self.last_interval_time
        if self.interval_func:
            self.interval_func(interval_time)
        self.last_interval_time = now
        return interval_time

    def stop(self):
        total_time = time.perf_counter() - self.start_time
        if self.stop_func:
            self.stop_func(total_time)
        return total_time


@contextmanager
def time_and_log_code_section(log_message: str, log_level=DEFAULT_LOG_LEVEL):
    timer = Timer(
        lambda elapsed: logger.log(
            log_level, f"{log_message} - interval {elapsed:.4f} seconds"
        ),
        lambda total: logger.log(
            log_level, f"{log_message} - total elapsed {total:.4f} seconds"
        ),
    )
    try:
        yield timer
    finally:
        timer.stop()

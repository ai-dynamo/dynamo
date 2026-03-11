# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from contextlib import contextmanager
from typing import Callable

logger = logging.getLogger(__name__)


DEFAULT_LOG_LEVEL = logging.INFO


class Timer:
    """Simple timer implementation that can time interval, on constructing the Timer,
    it starts the timer immediately, which will be stopped when calling stop().
    It also supports timing intervals by calling start_interval() and stop_interval(),
    so that you can time different parts of the code while the timer is running, note
    that stop_interval() will update the interval start time to current time.
    Example guide:
        t = Timer(
            lambda elapsed: logger.debug(f"phase1: {elapsed:.2f}s"),
            lambda total: logger.debug(f"total: {total:.2f}s"),
        )
        t.start_interval()
        do_phase1()
        t.stop_interval()  # prefer "start_something"
        do_phase2()
        t.stop()
    """

    def __init__(
        self,
        interval_func: Callable[[float], None] = None,
        stop_func: Callable[[float], None] = None,
    ):
        """Initialize the Timer and start timing immediately.

        Args:
            interval_func: Optional callback invoked with elapsed seconds during the timed interval when stop_interval() is called.
            stop_func: Optional callback invoked with total elapsed seconds when stop() is called.
        """
        self.start_time = time.perf_counter()
        self.interval_start_time = self.start_time
        self.interval_func = interval_func
        self.stop_func = stop_func

    def start_interval(self):
        """Start the interval timer."""
        self.interval_start_time = time.perf_counter()

    def stop_interval(self):
        """Stop the interval timer and return the elapsed time during the interval."""
        now = time.perf_counter()
        interval_time = now - self.interval_start_time
        if self.interval_func:
            self.interval_func(interval_time)
        self.interval_start_time = now
        return interval_time

    def stop(self):
        """Stop the timer and return the total elapsed time."""
        total_time = time.perf_counter() - self.start_time
        if self.stop_func:
            self.stop_func(total_time)
        return total_time


@contextmanager
def time_and_log_code_section(log_message: str, log_level=DEFAULT_LOG_LEVEL):
    """Context manager that times a code block and logs total elapsed on exit.

    Use the yielded timer's start_interval() and stop_interval() inside the block to
    log sub-intervals   (e.g. time to first token). Total elapsed is always logged
    when the block exits.

    Example:
        with time_and_log_code_section("[DECODE] generate") as t:
            t.start_interval()
            async for chunk in generator():
                if first_token:
                    t.stop_interval()  # Log time to first chunk
                    first_token = False
                yield chunk

    Expected output (at default log level DEBUG):
        [DECODE] generate - interval 0.1234 seconds   # if stop_interval() was called
        [DECODE] generate - total elapsed 1.5678 seconds   # always on exit

    Args:
        log_message: Base message to use for logging, interval and total times will be appended.
        log_level: Logging level to use for the messages (default: logging.DEBUG).
    """
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

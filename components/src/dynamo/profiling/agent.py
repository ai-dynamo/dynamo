# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight metadata/clock callback for the worker control process."""

import os
import socket
import time

from .collectors import process_epoch


class ProfilingAgent:
    def __init__(self, backend: str, worker_instance_id: int | None = None):
        self.backend = backend
        # The runtime connection ID, which the router records as the worker ID.
        self.worker_instance_id = worker_instance_id
        self.pid = os.getpid()
        self.epoch: str | None = None
        self.hostname = socket.gethostname()

    async def info(self, body: dict) -> dict:
        received_ns = time.time_ns()
        received_monotonic_ns = time.monotonic_ns()
        if self.epoch is None:
            self.epoch = process_epoch(self.pid)
        # This attests the control process, not every GPU rank in its group.
        return {
            "status": "ok",
            "schema_version": 1,
            "backend": self.backend,
            "pid": self.pid,
            "hostname": self.hostname,
            "process_epoch": self.epoch,
            "worker_instance_id": self.worker_instance_id,
            "scope": "control_process",
            "received_at_ns": received_ns,
            "sent_at_ns": time.time_ns(),
            "elapsed_monotonic_ns": time.monotonic_ns() - received_monotonic_ns,
        }

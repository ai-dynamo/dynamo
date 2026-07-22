# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-local vLLM worker extension used by RL discovery tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


class DynamoRLTestWorkerExtension(Worker):
    """Minimal extension for exercising RL admin collective RPC plumbing.

    Production RL integrations can provide real filesystem or NCCL weight loading.
    The Dynamo pytest suite only needs to verify that the worker admin route reaches
    a vLLM worker extension without depending on an external RL framework package.
    """

    def init_broadcaster(self, **_: object) -> None:
        return None

    def destroy_broadcaster(self, **_: object) -> None:
        return None

    def liveness_probe(self) -> None:
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        if not weight_path:
            raise ValueError("weight_path must be a non-empty string")
        return None

#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import os
import warnings
from typing import Any

from _bentoml_sdk import Service
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.resource import system_resources
from bentoml.exceptions import BentoMLConfigException
from simple_di import Provide, inject

NVIDIA_GPU = "nvidia.com/gpu"
DISABLE_GPU_ALLOCATION_ENV = "DYNAMO_DISABLE_GPU_ALLOCATION"
DYNAMO_DEPLOYMENT_ENV = "DYNAMO_DEPLOYMENT_ENV"


class ResourceAllocator:
    def __init__(self) -> None:
        self.system_resources = system_resources()
        self.remaining_gpus = len(self.system_resources[NVIDIA_GPU])
        self._available_gpus: list[tuple[float, float]] = [
            (1.0, 1.0)  # each item is (remaining, unit)
            for _ in range(self.remaining_gpus)
        ]

    def assign_gpus(self, count: float) -> list[int]:
        if count > self.remaining_gpus:
            warnings.warn(
                f"Requested {count} GPUs, but only {self.remaining_gpus} are remaining. "
                f"Serving may fail due to inadequate GPUs. Set {DISABLE_GPU_ALLOCATION_ENV}=1 "
                "to disable automatic allocation and allocate GPUs manually.",
                ResourceWarning,
                stacklevel=3,
            )
        self.remaining_gpus = int(max(0, self.remaining_gpus - count))
        if count < 1:  # a fractional GPU
            try:
                # try to find the GPU used with the same fragment
                gpu = next(
                    i
                    for i, v in enumerate(self._available_gpus)
                    if v[0] > 0 and v[1] == count
                )
            except StopIteration:
                try:
                    gpu = next(
                        i for i, v in enumerate(self._available_gpus) if v[0] == 1.0
                    )
                except StopIteration:
                    gpu = len(self._available_gpus)
                    self._available_gpus.append((1.0, count))
            remaining, _ = self._available_gpus[gpu]
            if (remaining := remaining - count) < count:
                # can't assign to the next one, mark it as zero.
                self._available_gpus[gpu] = (0.0, count)
            else:
                self._available_gpus[gpu] = (remaining, count)
            return [gpu]
        else:  # allocate n GPUs, n is a positive integer
            if int(count) != count:
                raise BentoMLConfigException(
                    "Float GPUs larger than 1 is not supported"
                )
            count = int(count)
            unassigned = [
                gpu
                for gpu, value in enumerate(self._available_gpus)
                if value[0] > 0 and value[1] == 1.0
            ]
            if len(unassigned) < count:
                warnings.warn(
                    f"Not enough GPUs to be assigned, {count} is requested",
                    ResourceWarning,
                )
                for _ in range(count - len(unassigned)):
                    unassigned.append(len(self._available_gpus))
                    self._available_gpus.append((1.0, 1.0))
            for gpu in unassigned[:count]:
                self._available_gpus[gpu] = (0.0, 1.0)
            return unassigned[:count]

    @inject
    def get_worker_env(
        self,
        service: Service[Any],
        services: dict[str, Any] = Provide[BentoMLContainer.config.services],
    ) -> tuple[int, list[dict[str, str]]]:
        config = services[service.name]

        num_gpus = 0
        num_workers = 1
        worker_env: list[dict[str, str]] = []
        if "gpu" in (config.get("resources") or {}):
            num_gpus = config["resources"]["gpu"]  # type: ignore
        if config.get("workers"):
            if (workers := config["workers"]) == "cpu_count":
                num_workers = int(self.system_resources["cpu"])
                # don't assign gpus to workers
                return num_workers, worker_env
            else:  # workers is a number
                num_workers = workers
        if num_gpus and DISABLE_GPU_ALLOCATION_ENV not in os.environ:
            if os.environ.get(DYNAMO_DEPLOYMENT_ENV):
                # K8s replicas: Assumes DYNAMO_DEPLOYMENT_ENV is set
                # each pod in replicaset will have separate GPU with same CUDA_VISIBLE_DEVICES
                assigned = self.assign_gpus(num_gpus)
                worker_env = [
                    {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                    for _ in range(num_workers)
                ]
            else:
                # local deployment where we split all available GPUs across workers
                for _ in range(num_workers):
                    assigned = self.assign_gpus(num_gpus)
                    worker_env.append(
                        {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                    )
        return num_workers, worker_env

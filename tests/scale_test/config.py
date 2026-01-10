# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_default_template_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "templates", "mocker_deployment.yaml")


@dataclass
class ScaleTestConfig:
    kubernetes_namespace: str = "default"
    template_path: Optional[str] = None
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    speedup_ratio: float = 10.0
    name_prefix: str = "scale-test"
    deployment_timeout: int = 600
    cleanup_on_exit: bool = True
    worker_replicas: int = 1

    def __post_init__(self):
        if self.template_path is None:
            self.template_path = _get_default_template_path()


@dataclass
class LoadTestConfig:
    duration_sec: int = 60
    qps: float = 1.0
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    num_pods: int = 1
    num_processes_per_pod: int = 1


@dataclass
class ScaleManagerConfig:
    num_deployments: int = 5
    scale_test: ScaleTestConfig = field(default_factory=ScaleTestConfig)
    load_test: LoadTestConfig = field(default_factory=LoadTestConfig)

    @classmethod
    def from_args(
        cls,
        num_deployments: int,
        namespace: str = "default",
        model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        speedup_ratio: float = 10.0,
        duration: int = 60,
        qps: float = 1.0,
        timeout: int = 600,
        load_gen_pods: int = 1,
        load_gen_processes: int = 1,
        name_prefix: str = "scale-test",
        cleanup_on_exit: bool = True,
        worker_replicas: int = 1,
    ) -> "ScaleManagerConfig":
        return cls(
            num_deployments=num_deployments,
            scale_test=ScaleTestConfig(
                kubernetes_namespace=namespace,
                model_path=model_path,
                speedup_ratio=speedup_ratio,
                deployment_timeout=timeout,
                name_prefix=name_prefix,
                cleanup_on_exit=cleanup_on_exit,
                worker_replicas=worker_replicas,
            ),
            load_test=LoadTestConfig(
                duration_sec=duration,
                qps=qps,
                model=model_path,
                num_pods=load_gen_pods,
                num_processes_per_pod=load_gen_processes,
            ),
        )

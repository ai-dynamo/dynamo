# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data models for sweep_core.

All data structures are plain dataclasses that serialize to/from JSON/dict.
No subprocess, kubectl, or argparse imports allowed in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

IsolationPolicy = Literal["fresh_per_run", "reuse_by_deploy_key"]


@dataclass(frozen=True)
class DeployKey:
    """Hashable key identifying a unique deployment configuration."""

    backend: str
    tokenizer: str
    workers: int
    num_models: int
    env_overrides: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "tokenizer": self.tokenizer,
            "workers": self.workers,
            "num_models": self.num_models,
            "env_overrides": dict(self.env_overrides),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DeployKey:
        env = d.get("env_overrides", {})
        return cls(
            backend=d["backend"],
            tokenizer=d["tokenizer"],
            workers=d["workers"],
            num_models=d["num_models"],
            env_overrides=frozenset(env.items())
            if isinstance(env, dict)
            else frozenset(env),
        )


@dataclass
class DeployDimension:
    """Configuration for a single deployment state."""

    backend: str  # "mocker" or "vllm"
    tokenizer: str  # "hf" or "fastokens"
    workers: int = 2
    num_models: int = 1
    env_overrides: Dict[str, str] = field(default_factory=dict)

    @property
    def deploy_key(self) -> DeployKey:
        return DeployKey(
            backend=self.backend,
            tokenizer=self.tokenizer,
            workers=self.workers,
            num_models=self.num_models,
            env_overrides=frozenset(self.env_overrides.items()),
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "tokenizer": self.tokenizer,
            "workers": self.workers,
            "num_models": self.num_models,
            "env_overrides": self.env_overrides,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DeployDimension:
        return cls(
            backend=d["backend"],
            tokenizer=d["tokenizer"],
            workers=d.get("workers", 2),
            num_models=d.get("num_models", 1),
            env_overrides=d.get("env_overrides", {}),
        )


@dataclass
class AiperfDimension:
    """Configuration for a single aiperf run."""

    concurrency: int
    isl: int
    osl: int = 256
    num_requests: Optional[int] = None
    benchmark_duration: Optional[int] = None
    request_rate: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "isl": self.isl,
            "osl": self.osl,
            "num_requests": self.num_requests,
            "benchmark_duration": self.benchmark_duration,
            "request_rate": self.request_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AiperfDimension:
        return cls(
            concurrency=d["concurrency"],
            isl=d["isl"],
            osl=d.get("osl", 256),
            num_requests=d.get("num_requests"),
            benchmark_duration=d.get("benchmark_duration"),
            request_rate=d.get("request_rate"),
        )


@dataclass
class RunSpec:
    """One logical perf run -- the atomic unit of execution."""

    deploy: DeployDimension
    aiperf: AiperfDimension
    deploy_key: DeployKey
    run_id: str

    def to_dict(self) -> dict:
        return {
            "deploy": self.deploy.to_dict(),
            "aiperf": self.aiperf.to_dict(),
            "deploy_key": self.deploy_key.to_dict(),
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunSpec:
        deploy = DeployDimension.from_dict(d["deploy"])
        aiperf = AiperfDimension.from_dict(d["aiperf"])
        deploy_key = DeployKey.from_dict(d["deploy_key"])
        return cls(
            deploy=deploy,
            aiperf=aiperf,
            deploy_key=deploy_key,
            run_id=d["run_id"],
        )


@dataclass
class RunResult:
    """Result from a single sweep point."""

    run_spec: RunSpec
    status: str = "pending"  # ok, fail, skipped
    req_per_sec: float = 0.0
    output_tok_per_sec: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p99_ms: float = 0.0
    duration_sec: float = 0.0
    run_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "run_spec": self.run_spec.to_dict(),
            "status": self.status,
            "req_per_sec": self.req_per_sec,
            "output_tok_per_sec": self.output_tok_per_sec,
            "ttft_p50_ms": self.ttft_p50_ms,
            "ttft_p99_ms": self.ttft_p99_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p99_ms": self.itl_p99_ms,
            "duration_sec": self.duration_sec,
            "run_dir": self.run_dir,
        }


@dataclass
class K8sConfig:
    """K8s-specific configuration."""

    namespace: str = "dynamo-bench"
    endpoint: str = "frontend:8000"
    dgd_name: str = ""
    image: str = ""
    frontend_port: int = 8000
    worker_replicas: int = 1
    frontend_replicas: int = 1
    deploy_template: str = ""  # path to deploy.yaml template
    reset_strategy: str = "graph"  # none | frontend | graph
    request_plane: str = "tcp"
    event_plane: str = "nats"
    router_mode: str = "round-robin"
    deploy: bool = False
    hf_token: str = ""
    image_pull_secret: str = ""
    export_level: str = "summary"

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "endpoint": self.endpoint,
            "dgd_name": self.dgd_name,
            "image": self.image,
            "frontend_port": self.frontend_port,
            "worker_replicas": self.worker_replicas,
            "frontend_replicas": self.frontend_replicas,
            "deploy_template": self.deploy_template,
            "reset_strategy": self.reset_strategy,
            "request_plane": self.request_plane,
            "event_plane": self.event_plane,
            "router_mode": self.router_mode,
            "deploy": self.deploy,
            "hf_token": "***" if self.hf_token else "",
            "image_pull_secret": self.image_pull_secret,
            "export_level": self.export_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> K8sConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SweepConfig:
    """Top-level configuration for a sweep."""

    model: str = "Qwen/Qwen3-0.6B"
    model_name: str = ""
    mode: str = "local"  # "local" or "k8s"
    backend: str = "mocker"
    tokenizers: List[str] = field(default_factory=lambda: ["hf", "fastokens"])
    concurrencies: List[int] = field(default_factory=lambda: [50, 100, 200])
    isls: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    osl: int = 256
    worker_counts: List[int] = field(default_factory=lambda: [2])
    num_models: int = 1
    aiperf_targets: str = "first"
    speedup_ratio: float = 1.0
    benchmark_duration: Optional[int] = 60
    num_requests_list: List[Optional[int]] = field(default_factory=lambda: [None])
    rps_list: List[Optional[int]] = field(default_factory=lambda: [None])
    output_dir: str = ""
    max_consecutive_fails: int = 2
    cooldown: int = 3
    dry_run: bool = False
    no_report: bool = False
    isolation_policy: IsolationPolicy = "fresh_per_run"
    passthrough_args: List[str] = field(default_factory=list)
    k8s: K8sConfig = field(default_factory=K8sConfig)

    def __post_init__(self):
        if not self.model_name:
            self.model_name = self.model

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "model_name": self.model_name,
            "mode": self.mode,
            "backend": self.backend,
            "tokenizers": self.tokenizers,
            "concurrencies": self.concurrencies,
            "isls": self.isls,
            "osl": self.osl,
            "worker_counts": self.worker_counts,
            "num_models": self.num_models,
            "aiperf_targets": self.aiperf_targets,
            "speedup_ratio": self.speedup_ratio,
            "benchmark_duration": self.benchmark_duration,
            "num_requests_list": self.num_requests_list,
            "rps_list": self.rps_list,
            "output_dir": self.output_dir,
            "max_consecutive_fails": self.max_consecutive_fails,
            "cooldown": self.cooldown,
            "dry_run": self.dry_run,
            "no_report": self.no_report,
            "isolation_policy": self.isolation_policy,
            "passthrough_args": self.passthrough_args,
            "k8s": self.k8s.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SweepConfig:
        k8s_data = d.pop("k8s", {})
        config = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if k8s_data:
            config.k8s = K8sConfig.from_dict(k8s_data)
        return config


@dataclass
class SweepPlan:
    """Serializable execution plan."""

    config: SweepConfig
    runs: List[RunSpec]
    isolation_policy: IsolationPolicy
    total_runs: int

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "isolation_policy": self.isolation_policy,
            "total_runs": self.total_runs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SweepPlan:
        config = SweepConfig.from_dict(d["config"])
        runs = [RunSpec.from_dict(r) for r in d["runs"]]
        return cls(
            config=config,
            runs=runs,
            isolation_policy=d["isolation_policy"],
            total_runs=d["total_runs"],
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> SweepPlan:
        return cls.from_dict(json.loads(s))

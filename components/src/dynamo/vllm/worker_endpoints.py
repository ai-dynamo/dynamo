# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role-scoped endpoint ownership for vLLM workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerEndpointSet:
    """All request/control endpoints published by one worker role.

    The primary and versioned aliases form one publication unit for sleep/wake.
    Control endpoints remain in shutdown membership but never participate in
    inference publication transactions.
    """

    primary: Any
    versioned: Any | None
    clear: Any
    perf: Any
    rl: Any | None = None
    lora: tuple[Any, ...] = ()

    @classmethod
    def decode(
        cls, runtime: Any, config: Any, *, versioned_path: str
    ) -> WorkerEndpointSet:
        prefix = f"{config.namespace}.{config.component}"
        versioned = (
            None if config.use_vllm_tokenizer else runtime.endpoint(versioned_path)
        )
        lora = (
            tuple(
                runtime.endpoint(f"{prefix}.{name}")
                for name in ("load_lora", "unload_lora", "list_loras")
            )
            if config.engine_args.enable_lora
            else ()
        )
        return cls(
            primary=runtime.endpoint(f"{prefix}.{config.endpoint}"),
            versioned=versioned,
            clear=runtime.endpoint(f"{prefix}.clear_kv_blocks"),
            perf=runtime.endpoint(f"{prefix}.get_perf_metrics"),
            rl=runtime.endpoint(f"{prefix}.rl") if config.enable_rl else None,
            lora=lora,
        )

    @classmethod
    def prefill(
        cls, runtime: Any, config: Any, *, versioned_path: str
    ) -> WorkerEndpointSet:
        prefix = f"{config.namespace}.{config.component}"
        return cls(
            primary=runtime.endpoint(f"{prefix}.{config.endpoint}"),
            versioned=runtime.endpoint(versioned_path),
            clear=runtime.endpoint(f"{prefix}.clear_kv_blocks"),
            perf=runtime.endpoint(f"{prefix}.get_perf_metrics"),
            rl=runtime.endpoint(f"{prefix}.rl") if config.enable_rl else None,
        )

    @property
    def routing_endpoints(self) -> tuple[Any, ...]:
        return (self.primary,) + (
            (self.versioned,) if self.versioned is not None else ()
        )

    @property
    def handler_args(self) -> dict[str, Any]:
        return {
            "generate_endpoint": self.primary,
            "additional_generate_endpoints": self.routing_endpoints[1:],
        }

    @property
    def shutdown_members(self) -> tuple[Any, ...]:
        optional = (self.rl,) if self.rl is not None else ()
        return (*self.routing_endpoints, self.clear, self.perf, *optional, *self.lora)

    def bind_shutdown(self, shutdown_endpoints: list[Any]) -> None:
        shutdown_endpoints[:] = self.shutdown_members

    def serve_tasks(
        self,
        handler: Any,
        *,
        metrics_labels: list[tuple[str, str]],
        health_check_payload: dict,
    ) -> list[Any]:
        generation_kwargs = {
            "graceful_shutdown": True,
            "metrics_labels": metrics_labels,
            "health_check_payload": health_check_payload,
        }
        tasks = [
            self.primary.serve_endpoint(handler.generate, **generation_kwargs),
            self.clear.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=metrics_labels,
            ),
            self.perf.serve_endpoint(
                handler.get_perf_metrics,
                metrics_labels=metrics_labels,
            ),
        ]
        if self.versioned is not None:
            tasks.append(
                self.versioned.serve_endpoint(handler.generate, **generation_kwargs)
            )
        if self.rl is not None:
            tasks.append(
                self.rl.serve_endpoint(
                    handler.rl_dispatch,
                    metrics_labels=metrics_labels,
                )
            )
        if self.lora:
            for endpoint, callback in zip(
                self.lora,
                (handler.load_lora, handler.unload_lora, handler.list_loras),
                strict=True,
            ):
                tasks.append(
                    endpoint.serve_endpoint(callback, metrics_labels=metrics_labels)
                )
        return tasks

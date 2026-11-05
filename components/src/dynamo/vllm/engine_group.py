# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-readiness-driven registration within the existing worker lifecycle."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from dynamo.llm.exceptions import HttpError

from .external_elastic_ep import observe_external_elastic_ep

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

    from dynamo.runtime import Client, Endpoint

logger = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    """Discovery I/O is inconclusive; it is not an engine failure."""


class NativeObservationError(RuntimeError):
    """Native serving state is inconclusive; registration must remain closed."""


@dataclass(frozen=True)
class NativeBinding:
    instance_id: str
    operation_id: str | None
    epoch: str | None
    data_parallel_size: int

    @classmethod
    def from_status(
        cls, status: dict, previous: NativeBinding | None = None
    ) -> NativeBinding | None:
        submission = status.get("local_submission")
        if (
            status["error"] is not None
            or not status["topology_known"]
            or (submission is not None and submission["error"] is not None)
        ):
            return None
        if status["phase"] == "preparing":
            # Preparation may preserve the old serving topology. Its new epoch
            # is not committed yet and cannot invalidate the old serving proof.
            if (
                previous is not None
                and previous.instance_id == status["instance_id"]
                and previous.data_parallel_size
                == status["effective_data_parallel_size"]
            ):
                return previous
            return None
        if status["phase"] not in {"idle", "completed"}:
            return None
        return cls(
            status["instance_id"],
            status["operation_id"],
            status["epoch"],
            status["effective_data_parallel_size"],
        )


class EngineGroupServing:
    """Observe native membership, verify progress, and maintain registration.

    No controller grant participates in this lifecycle. Native retirement keeps
    its control endpoint alive and unregistered until the allocation is released.
    """

    def __init__(
        self,
        engine: AsyncLLM,
        endpoint: Endpoint,
        pod_uid: str,
        rank: int,
        discovery: Client,
        can_serve: Callable[[], bool],
        verify: Callable[[], Awaitable[bool]],
        drain_requests: bool = False,
        rank_count: int = 1,
    ) -> None:
        if rank < 0 or rank_count < 1:
            raise ValueError("a worker must own a nonempty native DP range")
        self.engine = engine
        self.endpoint = endpoint
        self.pod_uid = pod_uid
        self.rank = rank
        self.rank_count = rank_count
        self.discovery = discovery
        self.can_serve = can_serve
        self.verify = verify
        self.drain_requests = drain_requests
        self._native_instance_id: str | None = None
        self._is_retired = False
        self._binding: NativeBinding | None = None
        self._verified: NativeBinding | None = None
        self._is_admitted = False
        self._is_ready = False
        self._is_registered: bool | None = False
        self._is_paused = False
        self._native_paused = True
        self._refresh_lock = asyncio.Lock()

    def listener_started(self) -> None:
        self._is_ready = True

    def listener_stopped(self) -> None:
        self._is_ready = False
        self._is_admitted = False

    def set_paused(self, paused: bool) -> None:
        self._is_paused = paused
        if paused:
            self._is_admitted = False

    def ensure_serving(self) -> None:
        if not self._is_admitted or not self.can_serve():
            raise HttpError(503, "Worker is temporarily unavailable")

    async def _observe_binding(self) -> None:
        try:
            native = await asyncio.wait_for(
                observe_external_elastic_ep(self.engine), timeout=5
            )
            self._native_instance_id = native["instance_id"]
            self._is_retired = (
                native["topology_known"]
                and native["phase"] == "completed"
                and self.rank >= native["effective_data_parallel_size"]
            )
            binding = NativeBinding.from_status(native, self._binding)
            # A retiring EngineCore may already be shut down; its native status remains queryable.
            self._native_paused = True
            if binding is not None and not self._is_retired and self.can_serve():
                self._native_paused = await asyncio.wait_for(
                    self.engine.is_paused(), timeout=5
                )
        except asyncio.CancelledError:
            self._is_admitted = False
            self._binding = None
            self._verified = None
            raise
        except Exception as error:
            self._is_admitted = False
            self._binding = self._verified = None
            self._native_instance_id = None
            self._is_retired = False
            raise NativeObservationError("native serving state is unknown") from error
        if (
            binding is not None
            and self.rank + self.rank_count > binding.data_parallel_size
        ):
            binding = None
        if binding != self._binding:
            self._verified = None
            self._is_admitted = False
        self._binding = binding
        if not self._eligible():
            self._is_admitted = False

    def _eligible(self) -> bool:
        return (
            self._is_ready
            and not self._is_paused
            and not self._native_paused
            and self._binding is not None
            and self.can_serve()
        )

    async def refresh(self) -> None:
        async with self._refresh_lock:
            await self._refresh()

    async def _refresh(self) -> None:
        await self._observe_binding()
        binding = self._binding
        if not self._eligible():
            await self.withdraw()
            return

        if self._verified != binding:
            # The runtime probes independently: no operation submission or
            # controller acknowledgment is needed to establish serving progress.
            await self.withdraw()
            completed = await self.verify()
            await self._observe_binding()
            if not completed or self._binding != binding or not self._eligible():
                await self.withdraw()
                return
            self._verified = binding

        if (
            self._is_registered is not True
            or self.endpoint.connection_id() not in self.discovery.instance_ids()
        ):
            await self._set_registered(True)
            await self._observe_binding()
            if self._binding != binding or not self._eligible():
                await self.withdraw()
                return
        self._is_admitted = True

    def status(self) -> dict:
        return {
            "pod_uid": self.pod_uid,
            "member_ids": [
                f"dp-{rank}" for rank in range(self.rank, self.rank + self.rank_count)
            ],
            "binding": asdict(self._binding) if self._binding else None,
            "verified": self._binding is not None
            and self._verified == self._binding
            and self._eligible(),
            "admitted": self._is_admitted and self._eligible(),
            "listener_ready": self._is_ready,
            "instance_id": self._native_instance_id,
            "retired": self._is_retired,
            "withdrawn": not self._is_admitted and self._is_registered is False,
            "native_drain_enabled": self.drain_requests,
        }

    async def observe(self, _body: dict) -> dict:
        await self._observe_binding()
        return self.status()

    async def maintain(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            try:
                await self.refresh()
            except (OSError, RegistrationError, NativeObservationError) as error:
                self._is_admitted = False
                logger.warning("Engine Group serving observation will retry: %s", error)
                try:
                    await self.withdraw()
                except RegistrationError:
                    logger.warning("Discovery withdrawal will retry")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=1)
            except TimeoutError:
                continue

    async def withdraw(self) -> None:
        self._is_admitted = False
        self._verified = None
        if (
            self._is_registered is not False
            or self.endpoint.connection_id() in self.discovery.instance_ids()
        ):
            await self._set_registered(False)

    async def _set_registered(self, registered: bool) -> None:
        self._is_admitted = False
        self._is_registered = None
        try:
            if registered:
                await self.endpoint.register_endpoint_instance()
            else:
                await self.endpoint.unregister_endpoint_instance()
        except Exception as error:
            logger.debug("Discovery registration outcome is unknown", exc_info=True)
            raise RegistrationError(
                "discovery registration outcome is unknown"
            ) from error
        self._is_registered = registered

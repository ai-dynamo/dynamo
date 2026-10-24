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
        if status["error"] is not None or not status["topology_known"]:
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
    ) -> None:
        self.engine = engine
        self.endpoint = endpoint
        self.pod_uid = pod_uid
        self.member_id = f"dp-{rank}"
        self.rank = rank
        self.discovery = discovery
        self.can_serve = can_serve
        self.verify = verify
        self.drain_requests = drain_requests
        self.native_instance_id: str | None = None
        self.is_retired = False
        self.binding: NativeBinding | None = None
        self.verified: NativeBinding | None = None
        self.is_admitted = False
        self.is_ready = False
        self.is_registered: bool | None = False
        self.is_paused = False
        self._native_paused = True
        self._refresh_lock = asyncio.Lock()

    def ensure_serving(self) -> None:
        if not self.is_admitted or not self.can_serve():
            raise HttpError(503, "Worker is temporarily unavailable")

    async def _observe_binding(self) -> None:
        try:
            native = await asyncio.wait_for(
                observe_external_elastic_ep(self.engine), timeout=5
            )
            self.native_instance_id = native["instance_id"]
            self.is_retired = (
                native["topology_known"]
                and native["phase"] == "completed"
                and self.rank >= native["effective_data_parallel_size"]
            )
            binding = NativeBinding.from_status(native, self.binding)
            # A retiring EngineCore may already be shut down; its native status remains queryable.
            self._native_paused = True
            if binding is not None and not self.is_retired and self.can_serve():
                self._native_paused = await asyncio.wait_for(
                    self.engine.is_paused(), timeout=5
                )
        except asyncio.CancelledError:
            self.is_admitted = False
            self.binding = None
            self.verified = None
            raise
        except Exception as error:
            self.is_admitted = False
            self.binding = self.verified = None
            self.native_instance_id = None
            self.is_retired = False
            raise NativeObservationError("native serving state is unknown") from error
        if binding is not None and self.rank >= binding.data_parallel_size:
            binding = None
        if binding != self.binding:
            self.verified = None
            self.is_admitted = False
        self.binding = binding
        if not self._eligible():
            self.is_admitted = False

    def _eligible(self) -> bool:
        return (
            self.is_ready
            and not self.is_paused
            and not self._native_paused
            and self.binding is not None
            and self.can_serve()
        )

    async def refresh(self) -> None:
        async with self._refresh_lock:
            await self._refresh()

    async def _refresh(self) -> None:
        await self._observe_binding()
        binding = self.binding
        if not self._eligible():
            await self.withdraw()
            return

        if self.verified != binding:
            # The runtime probes independently: no operation submission or
            # controller acknowledgment is needed to establish serving progress.
            await self.withdraw()
            completed = await self.verify()
            await self._observe_binding()
            if not completed or self.binding != binding or not self._eligible():
                await self.withdraw()
                return
            self.verified = binding

        if (
            self.is_registered is not True
            or self.endpoint.connection_id() not in self.discovery.instance_ids()
        ):
            await self._set_registered(True)
            await self._observe_binding()
            if self.binding != binding or not self._eligible():
                await self.withdraw()
                return
        self.is_admitted = True

    def status(self) -> dict:
        return {
            "pod_uid": self.pod_uid,
            "member_id": self.member_id,
            "binding": asdict(self.binding) if self.binding else None,
            "verified": self.binding is not None
            and self.verified == self.binding
            and self._eligible(),
            "admitted": self.is_admitted and self._eligible(),
            "listener_ready": self.is_ready,
            "instance_id": self.native_instance_id,
            "retired": self.is_retired,
            "withdrawn": not self.is_admitted and self.is_registered is False,
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
                self.is_admitted = False
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
        self.is_admitted = False
        self.verified = None
        if (
            self.is_registered is not False
            or self.endpoint.connection_id() in self.discovery.instance_ids()
        ):
            await self._set_registered(False)

    async def _set_registered(self, registered: bool) -> None:
        self.is_admitted = False
        self.is_registered = None
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
        self.is_registered = registered

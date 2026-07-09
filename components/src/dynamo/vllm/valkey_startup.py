# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional startup for Valkey-backed vLLM worker state."""

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

RegistrationT = TypeVar("RegistrationT")
PublisherT = TypeVar("PublisherT")


class UnsafeKvEventPublisherSetupError(RuntimeError):
    """Publisher setup failed without proving every partial publisher drained."""


class ValkeyStartupTarget(Protocol[RegistrationT, PublisherT]):
    valkey_worker_registration: RegistrationT | None
    kv_publishers: list[PublisherT] | None

    async def shutdown_valkey_worker_registration(self) -> None: ...

    def cleanup(self) -> None: ...


async def complete_valkey_startup(
    target: ValkeyStartupTarget[RegistrationT, PublisherT],
    *,
    acquire_registration: Callable[[], Awaitable[RegistrationT | None]],
    start_publishers: Callable[
        [RegistrationT | None], Awaitable[list[PublisherT] | None]
    ],
    register_model: Callable[[], Awaitable[None]],
) -> None:
    """Acquire the worker lease, publishers, and discovery atomically."""
    try:
        target.valkey_worker_registration = await acquire_registration()
        publishers = await start_publishers(target.valkey_worker_registration)
        if publishers:
            target.kv_publishers = publishers
        await register_model()
    except UnsafeKvEventPublisherSetupError:
        # A forced or timed-out partial publisher shutdown cannot prove that
        # no owner-fenced write remains. Stop renewal but let server lease
        # expiry fence late writes before ownership is reclaimed.
        target.valkey_worker_registration = None
        await target.shutdown_valkey_worker_registration()
        target.cleanup()
        raise
    except BaseException:
        await target.shutdown_valkey_worker_registration()
        target.cleanup()
        raise

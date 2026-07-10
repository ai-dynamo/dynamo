# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative vLLM scheduler, residency, and publication lifecycle state."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VllmEnginePauseController:
    """Own every state used to decide whether a worker may serve or mutate.

    vLLM's full ``wake_up`` resumes its scheduler, and ``is_sleeping`` reports
    scheduler pause *or* non-resident memory. Consequently neither a logical
    owner bit nor ``is_sleeping`` can certify a drained RL fence. Scheduler
    transitions are recorded as ambiguous before RPC dispatch and certified
    only by an acknowledged pause RPC; ``is_paused`` is used for reconciliation.
    """

    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._memory_sleeping = False
        self._memory_state_unknown = False
        self._generation_paused = False
        self._generation_pause_unknown = False
        self._generation_drained = False
        self._sleep_owner = False
        self._rl_owner = False
        self._rl_drained = False
        self._rl_mode = "wait"
        self._rl_clear_cache = False
        self._fatal_error: str | None = None
        self._routing_endpoints_published = True
        self._publication_recovery_needed = False

    @property
    def is_paused(self) -> bool:
        """Whether memory residency is asleep or indeterminate."""
        return self._memory_sleeping or self._memory_state_unknown

    @property
    def is_generation_paused(self) -> bool:
        return self._generation_paused or self._generation_pause_unknown

    @property
    def is_rl_paused(self) -> bool:
        return self._rl_owner

    @property
    def is_rl_drained(self) -> bool:
        return (
            self._fatal_error is None
            and not self.is_paused
            and self._rl_owner
            and self._rl_drained
            and self._generation_paused
            and not self._generation_pause_unknown
        )

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    @property
    def needs_resume_recovery(self) -> bool:
        return self.is_generation_paused

    @property
    def needs_sleep_recovery(self) -> bool:
        return self._memory_state_unknown or self._generation_pause_unknown or (
            self._sleep_owner and not self._memory_sleeping
        )

    @property
    def has_sleep_state(self) -> bool:
        return self._sleep_owner or self.is_paused

    @property
    def can_serve(self) -> bool:
        return (
            self._fatal_error is None
            and not self.is_paused
            and not self.is_generation_paused
            and not self._sleep_owner
            and not self._rl_owner
        )

    @property
    def routing_endpoints_published(self) -> bool:
        return self._routing_endpoints_published

    @property
    def publication_recovery_needed(self) -> bool:
        return self._publication_recovery_needed

    def mark_routing_publication(self, published: bool) -> None:
        self._routing_endpoints_published = published
        self._publication_recovery_needed = False

    def mark_routing_publication_failed(self) -> None:
        # A failed remote discovery write has an ambiguous outcome. Treat the
        # worker as unpublished locally and require a full convergence retry.
        self._routing_endpoints_published = False
        self._publication_recovery_needed = True

    def mark_fatal(self, message: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = message or "unknown post-mutation cache state"

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                f"engine state is indeterminate and requires restart: {self._fatal_error}"
            )

    async def _authoritative_pause_state(self) -> bool:
        is_paused = getattr(self._engine_client, "is_paused", None)
        if not callable(is_paused):
            raise RuntimeError("engine does not expose authoritative scheduler state")
        paused = await is_paused()
        if not isinstance(paused, bool):
            raise RuntimeError("engine returned a non-boolean scheduler state")
        return paused

    async def _reconcile_pause_after_error(self) -> bool | None:
        try:
            paused = await self._authoritative_pause_state()
        except BaseException:
            self._generation_pause_unknown = True
            logger.exception("Failed to reconcile vLLM scheduler pause outcome")
            return None

        self._generation_paused = paused
        # A positive state proves admission is closed, but it does not prove
        # that wait/abort draining completed after a lost RPC response.
        self._generation_pause_unknown = paused
        if not paused:
            self._generation_drained = False
        return paused

    async def _call_pause_generation(self, *, mode: str, clear_cache: bool) -> bool:
        """Pause and return whether the acknowledged call certifies draining."""
        try:
            await self._engine_client.pause_generation(
                mode=mode, clear_cache=clear_cache
            )
            return mode in ("wait", "abort")
        except TypeError:
            # Older vLLM accepts no keyword arguments. Reissuing is idempotent,
            # but the legacy completion cannot certify modern drain semantics.
            await self._engine_client.pause_generation()
            if clear_cache:
                reset_successful = await self._engine_client.reset_prefix_cache(
                    reset_connector=True
                )
                if reset_successful is not True:
                    raise RuntimeError(
                        "prefix/KV/connector cache reset did not complete"
                    )
            return False

    async def _establish_rl_pause(self) -> None:
        self._generation_pause_unknown = True
        self._generation_drained = False
        self._rl_drained = False
        try:
            drained = await self._call_pause_generation(
                mode=self._rl_mode,
                clear_cache=self._rl_clear_cache,
            )
        except BaseException:
            await self._reconcile_pause_after_error()
            raise

        self._generation_paused = True
        self._generation_pause_unknown = False
        self._generation_drained = drained
        self._rl_drained = drained

    async def _recover_failed_sleep(self) -> None:
        """Try an idempotent full wake; otherwise preserve fail-closed state."""
        self._memory_state_unknown = True
        try:
            paused = await self._authoritative_pause_state()
            if paused:
                await self._engine_client.wake_up()
                paused = await self._authoritative_pause_state()
        except BaseException:
            self._generation_pause_unknown = True
            logger.exception("Failed to reconcile native vLLM sleep outcome")
            return

        if paused:
            self._generation_paused = True
            self._generation_pause_unknown = True
            return

        self._memory_sleeping = False
        self._memory_state_unknown = False
        self._generation_paused = False
        self._generation_pause_unknown = False
        self._generation_drained = False
        if self._rl_owner:
            try:
                await self._establish_rl_pause()
            except BaseException:
                logger.exception("Failed to restore RL pause after sleep rollback")
                return
        self._sleep_owner = False

    async def pause(self, *args: object) -> bool:
        """Acquire sleep ownership before any scheduler or memory RPC."""
        self._raise_if_fatal()
        if self._sleep_owner or self.is_paused:
            return False

        level = args[0] if args else None
        self._sleep_owner = True
        self._memory_state_unknown = True

        if not self._generation_paused or self._generation_pause_unknown:
            self._generation_pause_unknown = True
            self._generation_drained = False
            try:
                await self._engine_client.pause_generation()
            except BaseException:
                paused = await self._reconcile_pause_after_error()
                if paused is False:
                    self._sleep_owner = False
                    self._memory_state_unknown = False
                raise
            self._generation_paused = True
            self._generation_pause_unknown = False
            self._generation_drained = True

        try:
            if level is None:
                await self._engine_client.sleep()
            else:
                await self._engine_client.sleep(level)
        except BaseException:
            await self._recover_failed_sleep()
            raise

        # An acknowledged vLLM sleep performs its own abort pause.
        self._memory_sleeping = True
        self._memory_state_unknown = False
        self._generation_paused = True
        self._generation_pause_unknown = False
        self._generation_drained = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        """Wake memory and restore any RL-owned drained scheduler fence."""
        self._raise_if_fatal()
        if not self.has_sleep_state:
            return False

        self._memory_state_unknown = True
        self._generation_pause_unknown = True
        try:
            if tags is None:
                await self._engine_client.wake_up()
            else:
                await self._engine_client.wake_up(tags)
            paused = await self._authoritative_pause_state()
        except BaseException:
            # The native wake may have completed. Keep both outcomes ambiguous
            # and force an idempotent retry before publication.
            raise

        if tags is not None and paused:
            self._memory_sleeping = True
            self._memory_state_unknown = False
            self._generation_paused = True
            self._generation_pause_unknown = False
            raise RuntimeError("engine is still sleeping after partial wake")

        self._memory_sleeping = False
        self._memory_state_unknown = False
        self._generation_paused = paused
        self._generation_pause_unknown = False
        self._generation_drained = False

        if self._rl_owner:
            await self._establish_rl_pause()
        elif paused:
            # Full wake should resume scheduling. An acknowledged but still
            # paused scheduler is not safe to publish without explicit recovery.
            self._generation_pause_unknown = True
            raise RuntimeError("scheduler remained paused after full wake")

        self._sleep_owner = False
        return True

    async def pause_generation(self, *, mode: str, clear_cache: bool) -> bool:
        """Acquire or recover the RL admission owner and its drained fence."""
        self._raise_if_fatal()
        changed = not self._rl_owner
        self._rl_owner = True
        self._rl_mode = mode
        self._rl_clear_cache = clear_cache

        if self._sleep_owner and self._generation_paused and not self._generation_pause_unknown:
            self._rl_drained = mode in ("wait", "abort") and self._generation_drained
            return changed
        if self.is_rl_drained:
            return False

        await self._establish_rl_pause()
        return changed

    async def resume_generation(self) -> bool:
        """Release RL ownership without treating a lost response as drained."""
        self._raise_if_fatal()
        if not self._rl_owner:
            return False

        self._rl_drained = False
        if self._sleep_owner:
            self._rl_owner = False
            return True

        self._generation_pause_unknown = True
        self._generation_drained = False
        try:
            await self._engine_client.resume_generation()
        except BaseException:
            await self._reconcile_pause_after_error()
            raise

        self._generation_paused = False
        self._generation_pause_unknown = False
        self._rl_owner = False
        return True

    def mark_resumed(self) -> None:
        """Compatibility no-op; successful RPCs commit state internally."""
        return None

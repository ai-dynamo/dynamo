# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe helpers for vLLM LoRA lifecycle mutations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo
from dynamo.common.utils.env import env_bool

logger = logging.getLogger(__name__)


async def run_lora_mutation(
    operation: Awaitable[Any],
) -> tuple[bool, Any | None, BaseException | None]:
    """Finish a state mutation even when its caller is cancelled."""
    task = asyncio.ensure_future(operation)
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_requested = True
        except Exception:  # noqa: BLE001,S110 - inspect the task below
            pass
    try:
        result = task.result()
    except (Exception, asyncio.CancelledError) as exc:  # mutation trust boundary
        return cancellation_requested, None, exc
    return cancellation_requested, result, None


class AdminLoRALifecycleOwner(Protocol):
    """Handler operations used by one administrative LoRA load transaction."""

    _lora_state: Any
    _engine_loaded_loras: set[str]
    _lora_capacity: int | None
    engine_client: Any

    async def _reconcile_uncertain_lora_discovery(
        self, lora_name: str, old_info: LoRAInfo | None
    ) -> tuple[bool, BaseException | None]:
        ...

    async def _reserve_admin_lora_slot(
        self, lora_name: str, lora_capacity: int | None
    ) -> int | None:
        ...

    async def _resolve_lora_source_path(self, lora_uri: str) -> tuple[bool, str]:
        ...

    async def _rollback_admin_lora_engine(
        self,
        lora_name: str,
        old_info: LoRAInfo | None,
        *,
        old_engine_loaded: bool,
        remove_lora_id: int | None,
    ) -> tuple[bool, BaseException | None]:
        ...

    async def _register_lora_discovery(self, lora_name: str, lora_id: int) -> None:
        ...

    async def _unregister_lora_discovery(self, lora_name: str) -> None:
        ...

    def _preload_lora_into_engine(self) -> bool:
        ...

    def _is_lora_not_loaded_error(self, error: Exception) -> bool:
        ...


@dataclass(frozen=True)
class AdminLoRALoadResult:
    """Semantic result rendered by the handler's streaming admin endpoint."""

    status: str
    message: str
    lora_name: str | None = None
    lora_id: int | None = None
    hot_swap: bool | None = None


class AdminLoRALoadTransaction:
    """Own one cancellation-safe administrative LoRA load or hot swap."""

    def __init__(
        self,
        owner: AdminLoRALifecycleOwner,
        lora_name: str,
        lora_uri: str,
    ) -> None:
        self.owner = owner
        self.lora_name = lora_name
        self.lora_uri = lora_uri
        self.old_info: LoRAInfo | None = None
        self.lora_id = -1
        self.is_hot_swap = False
        self.old_engine_loaded = False
        self.preload_into_engine = False
        self.capacity_reserved = False
        self.committed_lora_info = False

    def _result(
        self,
        status: str,
        message: str,
        *,
        include_name: bool = True,
        include_id: bool = False,
        hot_swap: bool | None = None,
    ) -> AdminLoRALoadResult:
        return AdminLoRALoadResult(
            status=status,
            message=message,
            lora_name=self.lora_name if include_name else None,
            lora_id=self.lora_id if include_id else None,
            hot_swap=hot_swap,
        )

    async def run(self) -> AdminLoRALoadResult:
        try:
            self.old_info = self.owner._lora_state.loaded_loras.get(self.lora_name)
            if self.old_info is not None:
                self.lora_id = self.old_info.id
            if self.owner._lora_state.is_runtime_managed(self.lora_name):
                return self._result(
                    "error",
                    "Request-time LoRA adapters cannot be replaced through the admin endpoint",
                    include_name=False,
                )

            self.is_hot_swap = self.old_info is not None and env_bool(
                "DYN_LORA_HOTSWAP_ENABLED"
            )
            self.old_engine_loaded = self.lora_name in self.owner._engine_loaded_loras

            result = await self._reconcile_discovery()
            if result is not None:
                return result
            if self.old_info is not None and not self.is_hot_swap:
                logger.info(
                    "LoRA adapter already loaded: %s with ID %s",
                    self.lora_name,
                    self.old_info.id,
                )
                return self._result(
                    "success",
                    f"LoRA adapter '{self.lora_name}' already loaded",
                    include_id=True,
                    hot_swap=False,
                )

            result = await self._reserve_capacity()
            if result is not None:
                return result
            result, lora_path = await self._resolve_source_path()
            if result is not None:
                return result

            result = await self._remove_old_adapter()
            if result is not None:
                return result
            self.preload_into_engine = (
                self.owner._preload_lora_into_engine() or self.is_hot_swap
            )
            result = await self._add_candidate(lora_path)
            if result is not None:
                return result

            self._commit_candidate(lora_path)
            result = await self._reset_prefix_cache()
            if result is not None:
                return result
            result = await self._publish_discovery()
            if result is not None:
                return result

            return self._result(
                "success",
                f"LoRA adapter '{self.lora_name}' "
                f"{'hot-swapped' if self.is_hot_swap else 'loaded'} successfully",
                include_id=True,
                hot_swap=self.is_hot_swap,
            )
        except Exception as error:  # noqa: BLE001 - admin endpoint error boundary
            if self.capacity_reserved:
                self.owner._lora_state.loaded_loras.pop(self.lora_name, None)
            logger.exception("Failed to load LoRA adapter: %s", error)
            return self._result("error", str(error), include_name=False)
        finally:
            self._release_reservation()

    async def _reconcile_discovery(self) -> AdminLoRALoadResult | None:
        cancelled, error = await self.owner._reconcile_uncertain_lora_discovery(
            self.lora_name, self.old_info
        )
        if error is not None:
            if cancelled:
                raise asyncio.CancelledError
            return self._result(
                "error",
                f"Discovery state for LoRA '{self.lora_name}' "
                f"could not be reconciled: {error}",
            )
        if cancelled:
            raise asyncio.CancelledError
        return None

    async def _reserve_capacity(self) -> AdminLoRALoadResult | None:
        if self.old_info is not None:
            return None
        lora_capacity = getattr(self.owner, "_lora_capacity", None)
        reserved_id = await self.owner._reserve_admin_lora_slot(
            self.lora_name, lora_capacity
        )
        if reserved_id is None:
            return self._result(
                "error",
                "LoRA capacity exceeded: "
                f"at most {lora_capacity} adapter(s) may be loaded",
            )
        self.lora_id = reserved_id
        self.capacity_reserved = True
        return None

    async def _resolve_source_path(
        self,
    ) -> tuple[AdminLoRALoadResult | None, str]:
        logger.info(
            "Downloading LoRA adapter: %s from %s", self.lora_name, self.lora_uri
        )
        path_ok, path_or_error = await self.owner._resolve_lora_source_path(
            self.lora_uri
        )
        if not path_ok:
            if self.capacity_reserved:
                self.owner._lora_state.loaded_loras.pop(self.lora_name, None)
            return (
                self._result("error", path_or_error, include_name=False),
                "",
            )
        logger.debug("LoRA downloaded to: %s", path_or_error)
        return None, path_or_error

    async def _remove_old_adapter(self) -> AdminLoRALoadResult | None:
        if not (
            self.is_hot_swap and self.old_info is not None and self.old_engine_loaded
        ):
            return None

        remove_cancelled, _removed, remove_error = await run_lora_mutation(
            self.owner.engine_client.remove_lora(self.old_info.id)
        )
        if remove_error is not None and not (
            isinstance(remove_error, Exception)
            and self.owner._is_lora_not_loaded_error(remove_error)
        ):
            (
                rollback_cancelled,
                rollback_error,
            ) = await self.owner._rollback_admin_lora_engine(
                self.lora_name,
                self.old_info,
                old_engine_loaded=True,
                remove_lora_id=None,
            )
            if rollback_error is not None:
                logger.error(
                    "Failed to restore LoRA %s after remove error: %s",
                    self.lora_name,
                    rollback_error,
                )
            logger.error(
                "Failed to remove existing LoRA '%s' before hot-swap: %s",
                self.lora_name,
                remove_error,
            )
            if remove_cancelled or rollback_cancelled:
                raise asyncio.CancelledError
            return self._result(
                "error",
                f"Failed to remove existing LoRA '{self.lora_name}' "
                f"before hot-swap: {remove_error}",
            )

        self.owner._engine_loaded_loras.discard(self.lora_name)
        if remove_cancelled:
            (
                rollback_cancelled,
                rollback_error,
            ) = await self.owner._rollback_admin_lora_engine(
                self.lora_name,
                self.old_info,
                old_engine_loaded=True,
                remove_lora_id=None,
            )
            if rollback_error is not None:
                logger.error(
                    "Failed to restore LoRA %s after cancellation: %s",
                    self.lora_name,
                    rollback_error,
                )
            if remove_cancelled or rollback_cancelled:
                raise asyncio.CancelledError
        return None

    async def _add_candidate(self, lora_path: str) -> AdminLoRALoadResult | None:
        if not self.preload_into_engine:
            return None
        add_cancelled, added, add_error = await run_lora_mutation(
            self.owner.engine_client.add_lora(
                LoRARequest(
                    lora_name=self.lora_name,
                    lora_int_id=self.lora_id,
                    lora_path=lora_path,
                )
            )
        )
        if not (add_cancelled or add_error is not None or added is not True):
            self.owner._engine_loaded_loras.add(self.lora_name)
            return None

        (
            rollback_cancelled,
            rollback_error,
        ) = await self.owner._rollback_admin_lora_engine(
            self.lora_name,
            self.old_info,
            old_engine_loaded=bool(self.is_hot_swap and self.old_engine_loaded),
            remove_lora_id=(
                self.lora_id if added is True or add_error is not None else None
            ),
        )
        if rollback_error is not None:
            logger.error(
                "Rollback failed for LoRA %s: %s", self.lora_name, rollback_error
            )
        if add_cancelled or rollback_cancelled:
            raise asyncio.CancelledError
        failure = add_error or RuntimeError("vLLM rejected the LoRA adapter")
        return self._result(
            "error",
            f"Failed to add LoRA '{self.lora_name}': {failure}",
        )

    def _commit_candidate(self, lora_path: str) -> None:
        self.owner._lora_state.admin_reserved_ids.pop(self.lora_name, None)
        self.owner._lora_state.loaded_loras[self.lora_name] = LoRAInfo(
            id=self.lora_id, path=lora_path
        )
        self.committed_lora_info = True
        logger.info(
            "Successfully %s LoRA adapter: %s with ID %s",
            "hot-swapped" if self.is_hot_swap else "loaded",
            self.lora_name,
            self.lora_id,
        )

    async def _reset_prefix_cache(self) -> AdminLoRALoadResult | None:
        if not self.is_hot_swap:
            return None
        reset_cancelled, reset, reset_error = await run_lora_mutation(
            self.owner.engine_client.reset_prefix_cache()
        )
        if not (reset_cancelled or reset_error is not None or reset is not True):
            return None

        rolled_back = "tracking only"
        if self.old_info is not None:
            (
                rollback_cancelled,
                rollback_error,
            ) = await self.owner._rollback_admin_lora_engine(
                self.lora_name,
                self.old_info,
                old_engine_loaded=self.old_engine_loaded,
                remove_lora_id=(self.lora_id if self.preload_into_engine else None),
            )
            if rollback_error is None:
                rolled_back = (
                    "engine+tracking" if self.old_engine_loaded else "tracking only"
                )
            else:
                logger.error(
                    "LoRA '%s' hot-swap engine rollback failed: %s",
                    self.lora_name,
                    rollback_error,
                )
        else:
            rollback_cancelled = False
            self.owner._lora_state.loaded_loras.pop(self.lora_name, None)
        if reset_cancelled or rollback_cancelled:
            raise asyncio.CancelledError
        failure = reset_error or RuntimeError("vLLM rejected the prefix cache reset")
        logger.error(
            "LoRA '%s' hot-swap rolled back (%s): prefix cache reset failed: %s",
            self.lora_name,
            rolled_back,
            failure,
        )
        return self._result(
            "error",
            f"LoRA '{self.lora_name}' hot-swap aborted; prefix cache reset failed: {failure}",
            include_id=True,
        )

    async def _publish_discovery(self) -> AdminLoRALoadResult | None:
        if self.is_hot_swap:
            return None
        (
            registration_cancelled,
            _registered,
            registration_error,
        ) = await run_lora_mutation(
            self.owner._register_lora_discovery(self.lora_name, self.lora_id)
        )
        if registration_error is None:
            self.owner._lora_state.discovery_uncertain_loras.discard(self.lora_name)
            logger.info(
                "Successfully published LoRA '%s' ModelDeploymentCard", self.lora_name
            )
            if registration_cancelled:
                raise asyncio.CancelledError
            return None

        logger.error(
            "Failed to publish LoRA %s ModelDeploymentCard: %s",
            self.lora_name,
            registration_error,
        )
        unregister_cancelled, _unregistered, unregister_error = await run_lora_mutation(
            self.owner._unregister_lora_discovery(self.lora_name)
        )
        if unregister_error is not None:
            self.owner._lora_state.discovery_uncertain_loras.add(self.lora_name)
            logger.error(
                "Failed to reconcile discovery for LoRA %s: %s",
                self.lora_name,
                unregister_error,
            )
            if registration_cancelled or unregister_cancelled:
                raise asyncio.CancelledError
            return self._result(
                "error",
                f"Failed to register LoRA '{self.lora_name}' and "
                "could not confirm discovery cleanup",
            )

        self.owner._lora_state.discovery_uncertain_loras.discard(self.lora_name)
        (
            rollback_cancelled,
            rollback_error,
        ) = await self.owner._rollback_admin_lora_engine(
            self.lora_name,
            None,
            old_engine_loaded=False,
            remove_lora_id=(self.lora_id if self.preload_into_engine else None),
        )
        if rollback_error is not None:
            logger.error(
                "Failed to rollback LoRA %s: %s", self.lora_name, rollback_error
            )
        if registration_cancelled or unregister_cancelled or rollback_cancelled:
            raise asyncio.CancelledError
        return self._result(
            "error",
            f"Failed to register LoRA '{self.lora_name}' in "
            f"discovery registry: {registration_error}",
        )

    def _release_reservation(self) -> None:
        self.owner._lora_state.admin_reserved_ids.pop(self.lora_name, None)
        if self.capacity_reserved and not self.committed_lora_info:
            existing = self.owner._lora_state.loaded_loras.get(self.lora_name)
            if existing is not None and existing.id == -1:
                self.owner._lora_state.loaded_loras.pop(self.lora_name, None)

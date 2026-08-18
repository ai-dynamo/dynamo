# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA adapter machinery for vLLM worker handlers.

Both worker families serve LoRA adapters, but they do not share a base class:
``BaseWorkerHandler`` drives generation, while ``EmbeddingWorkerHandler`` is a
plain class because the generation-only initialization (media loaders, KV block
lookup, embedding cache) is meaningless on a pooling engine. Keeping the
adapter lifecycle here lets both inherit it without inheriting the other's
request path.

Hosts must provide the attributes set up by :meth:`LoRAHandlerMixin.init_lora_state`
plus ``config``, ``engine_args``, ``engine_client``, and ``generate_endpoint``,
and must implement :meth:`_register_lora_discovery` — adapter cards carry the
host's model type, which differs between generation and pooling roles.
"""

import asyncio
import functools
import logging
from typing import Any

from vllm.lora.request import LoRARequest

from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.common.rl import (
    RLAdminValidationError,
    env_bool,
    require_lora_load_request,
    require_lora_unload_request,
)
from dynamo.llm import lora_name_to_id, unregister_model
from dynamo.runtime.logging import configure_dynamo_logging

from .constants import DisaggregationMode
from .lora_state import LoRAState

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class LoRAHandlerMixin:
    """Adapter resolution, lifecycle endpoints, and discovery for LoRA hosts."""

    def init_lora_state(self, config, generate_endpoint=None) -> None:
        """Initialize adapter tracking. Call from the host's ``__init__``."""
        self.config = config
        self.engine_args = config.engine_args
        self.generate_endpoint = generate_endpoint
        self._served_model_name = config.served_model_name or config.model
        self._served_model_aliases = tuple(config.served_model_aliases or ())
        self._lora_state = LoRAState()
        # Adapters known to have been handed to vLLM. Prefill registration is
        # metadata-only, but vLLM activates a prefill adapter lazily when an
        # inference request supplies its LoRARequest.
        self._engine_loaded_loras: set[str] = set()
        # Shared lock protecting capacity check and insertion into loaded_loras.
        # Per-adapter locks (via _get_lora_lock) serialize ops on the same adapter,
        # but concurrent loads of *different* adapters need a shared capacity guard
        # to prevent both bypassing the check before either inserts (atomicity).
        self._lora_capacity_guard = asyncio.Lock()

    async def _register_lora_discovery(self, lora_name: str, lora_id: int) -> None:
        """Publish a loaded adapter to discovery.

        Implemented per role: adapter cards inherit the host's model type, which
        differs between the generation and pooling families.
        """
        raise NotImplementedError

    @property
    def loaded_loras(self) -> dict[str, LoRAInfo]:
        """Compatibility alias for LoRAState-backed adapter tracking."""
        return self._lora_state.loaded_loras

    @loaded_loras.setter
    def loaded_loras(self, value: dict[str, LoRAInfo]) -> None:
        self._lora_state.loaded_loras = value

    @functools.cached_property
    def _lora_enabled(self) -> bool:
        """Conservative default for handlers that don't override LoRA policy.

        LoRA is considered enabled only when the engine args flag is set and a
        LoRA manager is available.
        """
        enable_lora = bool(getattr(self.engine_args, "enable_lora", False))
        return enable_lora and (get_lora_manager() is not None)

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest for loaded adapters, or None for base model names.

        Raises ValueError for unknown non-base names when LoRA is enabled.

        **Contract for subclasses**: This method requires the following attributes
        to be defined by subclasses:
        - `self._served_model_name` (str): The model name served by this worker
        - `self._served_model_aliases` (tuple[str, ...]): Additional served-model aliases that should resolve to the base model
        - `self.engine_args.model` (str): The base model path/name from engine args
        - `self._lora_enabled()` (method): Returns bool indicating if LoRA is enabled

        Subclasses that forget to define these will get AttributeError at runtime
        when this method is called. The concrete decode, prefill, and Omni handlers
        provide examples.
        """
        return self._lora_state.resolve_request(
            model_name,
            base_model_names=(
                self._served_model_name,
                self.engine_args.model,
                *self._served_model_aliases,
            ),
            lora_enabled=self._lora_enabled,
        )

    def _track_lora_request_activation(self, lora_request: LoRARequest | None) -> None:
        """Record adapters handed to vLLM for request-time lazy activation."""
        if lora_request is not None:
            self._engine_loaded_loras.add(lora_request.lora_name)

    @staticmethod
    def _is_lora_not_loaded_error(error: Exception) -> bool:
        """Return whether vLLM reports an idempotent remove of a missing LoRA."""
        message = str(error).lower()
        return "not loaded" in message or "not found" in message

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Get/create the per-LoRA lock without eagerly allocating a new lock each call."""
        return self._lora_state.get_lock(lora_name)

    def _parse_lora_unload_request(self, request: Any) -> str:
        """Parse and validate a LoRA unload request payload."""
        return require_lora_unload_request(request)

    async def _resolve_lora_source_path(self, lora_uri: str) -> tuple[bool, str]:
        """Resolve a LoRA URI into a local filesystem path.

        Returns:
            (ok, value): on success, (True, local_path); on failure,
            (False, error_message).
        """
        lora_manager = get_lora_manager()
        if lora_manager is None:
            return (
                False,
                "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
            )

        download_result = await lora_manager.download_lora(lora_uri)
        if download_result["status"] != "success":
            return (
                False,
                f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
            )

        return True, download_result["local_path"]

    async def _unregister_lora_discovery(self, lora_name: str) -> None:
        """Remove a loaded LoRA adapter from discovery."""
        if self.generate_endpoint is None:
            logger.debug(
                "Cannot unregister LoRA '%s': generate_endpoint=%s",
                lora_name,
                self.generate_endpoint,
            )
            return
        await unregister_model(
            endpoint=self.generate_endpoint,
            lora_name=lora_name,
        )

    def _preload_lora_into_engine(self) -> bool:
        """Whether lifecycle registration should eagerly activate the adapter.

        Prefill keeps the downloaded adapter metadata and supplies its path in
        the inference-time ``LoRARequest``. Decode and aggregated workers must
        be immediately ready to generate and therefore continue to preload.
        """
        return self.config.disaggregation_mode != DisaggregationMode.PREFILL

    async def load_lora(self, request=None):
        """
        Load a LoRA adapter dynamically into the vLLM's AsyncLLM engine.

        Request format:
        {
            "lora_name": str,
            "source": {
                "uri": str  # e.g., "s3://bucket/path" or "file:///path"
            }
        }

        Concurrent calls for the same LoRA are serialized. Re-loading an already
        loaded LoRA is idempotent by default. Set
        ``DYN_LORA_HOTSWAP_ENABLED=true`` to replace an already loaded LoRA with
        a new URI.
        """
        try:
            try:
                lora_name, lora_uri = require_lora_load_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return

            # Debug: Log the incoming request
            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                capacity_reserved = False
                committed_lora_info = False
                try:
                    old_info = self._lora_state.loaded_loras.get(lora_name)
                    hot_swap_enabled = env_bool("DYN_LORA_HOTSWAP_ENABLED")
                    is_hot_swap = old_info is not None and hot_swap_enabled
                    old_engine_loaded = lora_name in self._engine_loaded_loras

                    if old_info is not None and not hot_swap_enabled:
                        logger.info(
                            f"LoRA adapter already loaded: {lora_name} "
                            f"with ID {old_info.id}"
                        )
                        yield {
                            "status": "success",
                            "message": f"LoRA adapter '{lora_name}' already loaded",
                            "lora_name": lora_name,
                            "lora_id": old_info.id,
                            "hot_swap": False,
                        }
                        return

                    lora_capacity = getattr(self, "_lora_capacity", None)
                    # Guard capacity check: serialize new adapter loads to prevent two
                    # concurrent loads from both observing capacity below limit and proceeding.
                    if lora_capacity is not None and old_info is None:
                        async with self._lora_capacity_guard:
                            # Re-check under lock in case another load slipped in
                            if len(self._lora_state.loaded_loras) >= lora_capacity:
                                yield {
                                    "status": "error",
                                    "message": (
                                        "LoRA capacity exceeded: "
                                        f"at most {lora_capacity} adapter(s) may be loaded"
                                    ),
                                    "lora_name": lora_name,
                                }
                                return
                            # Reserve a capacity slot with placeholder (will be replaced below).
                            self._lora_state.loaded_loras[lora_name] = LoRAInfo(
                                id=-1, path=""
                            )
                            capacity_reserved = True

                    logger.info(
                        f"Downloading LoRA adapter: {lora_name} from {lora_uri}"
                    )
                    path_ok, lora_path_or_error = await self._resolve_lora_source_path(
                        lora_uri
                    )
                    if not path_ok:
                        if capacity_reserved:
                            self._lora_state.loaded_loras.pop(lora_name, None)
                        yield {
                            "status": "error",
                            "message": lora_path_or_error,
                        }
                        return

                    lora_path = lora_path_or_error
                    logger.debug(f"LoRA downloaded to: {lora_path}")

                    # Generate deterministic ID from lora_name before using it
                    lora_id = lora_name_to_id(lora_name)

                    if is_hot_swap and old_info is not None and old_engine_loaded:
                        try:
                            await self.engine_client.remove_lora(old_info.id)
                            self._engine_loaded_loras.discard(lora_name)
                        except Exception as e:
                            if capacity_reserved:
                                self._lora_state.loaded_loras.pop(lora_name, None)
                            logger.error(
                                f"Failed to remove existing LoRA '{lora_name}' "
                                f"before hot-swap: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"Failed to remove existing LoRA '{lora_name}' "
                                    f"before hot-swap: {e}"
                                ),
                                "lora_name": lora_name,
                            }
                            return

                    # Initial prefill registration is metadata-only. A hot
                    # swap must still replace any lazily activated old adapter
                    # atomically before the prefix cache is reset.
                    preload_into_engine = (
                        self._preload_lora_into_engine() or is_hot_swap
                    )
                    if preload_into_engine:
                        try:
                            await self.engine_client.add_lora(
                                LoRARequest(
                                    lora_name=lora_name,
                                    lora_int_id=lora_id,
                                    lora_path=lora_path,
                                )
                            )
                            self._engine_loaded_loras.add(lora_name)
                        except Exception as e:
                            if (
                                is_hot_swap
                                and old_info is not None
                                and old_engine_loaded
                            ):
                                try:
                                    await self.engine_client.add_lora(
                                        LoRARequest(
                                            lora_name=lora_name,
                                            lora_int_id=old_info.id,
                                            lora_path=old_info.path,
                                        )
                                    )
                                    self._engine_loaded_loras.add(lora_name)
                                except Exception as rollback_error:
                                    self._lora_state.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"Rollback failed for LoRA {lora_name}: "
                                        f"{rollback_error}"
                                    )
                            else:
                                # For new loads that weren't hot-swap, clean up reservation
                                if capacity_reserved:
                                    self._lora_state.loaded_loras.pop(lora_name, None)
                            yield {
                                "status": "error",
                                "message": f"Failed to add LoRA '{lora_name}': {e}",
                                "lora_name": lora_name,
                            }
                            return

                    # Insert or update the real LoRA info (replaces placeholder if reserved).
                    self._lora_state.loaded_loras[lora_name] = LoRAInfo(
                        id=lora_id, path=lora_path
                    )
                    committed_lora_info = True
                    logger.info(
                        f"Successfully {'hot-swapped' if is_hot_swap else 'loaded'} "
                        f"LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    if is_hot_swap:
                        try:
                            await self.engine_client.reset_prefix_cache()
                        except Exception as e:
                            # The new adapter is already active in the engine, but
                            # the prefix cache still holds entries computed under
                            # the old adapter and could be reused incorrectly.
                            # Roll the ENGINE back to old_info (remove new, re-add
                            # old) so engine state and our tracking stay consistent
                            # — a metadata-only rollback would leave the new adapter
                            # live while we report/route the old one (codex).
                            rolled_back = "tracking only"
                            if old_info is not None:
                                try:
                                    if preload_into_engine:
                                        await self.engine_client.remove_lora(lora_id)
                                        self._engine_loaded_loras.discard(lora_name)
                                    if old_engine_loaded:
                                        await self.engine_client.add_lora(
                                            LoRARequest(
                                                lora_name=lora_name,
                                                lora_int_id=old_info.id,
                                                lora_path=old_info.path,
                                            )
                                        )
                                        self._engine_loaded_loras.add(lora_name)
                                    self._lora_state.loaded_loras[lora_name] = old_info
                                    rolled_back = (
                                        "engine+tracking"
                                        if old_engine_loaded
                                        else "tracking only"
                                    )
                                except Exception as rollback_error:
                                    # Engine is in an indeterminate adapter state;
                                    # drop tracking so we never claim a clean swap.
                                    self._lora_state.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"LoRA '{lora_name}' hot-swap engine "
                                        f"rollback failed: {rollback_error}"
                                    )
                            else:
                                self._lora_state.loaded_loras.pop(lora_name, None)
                            logger.error(
                                f"LoRA '{lora_name}' hot-swap rolled back "
                                f"({rolled_back}): prefix cache reset failed: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"LoRA '{lora_name}' hot-swap aborted; prefix "
                                    f"cache reset failed: {e}"
                                ),
                                "lora_name": lora_name,
                                "lora_id": lora_id,
                            }
                            return

                    if not is_hot_swap:
                        try:
                            await self._register_lora_discovery(lora_name, lora_id)
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Roll back engine state when this worker preloaded;
                            # prefill only needs to discard the cached metadata.
                            try:
                                if preload_into_engine:
                                    logger.debug(
                                        f"Rolling back: removing LoRA '{lora_name}' from engine"
                                    )
                                    await self.engine_client.remove_lora(lora_id)
                                    self._engine_loaded_loras.discard(lora_name)
                                self._lora_state.loaded_loras.pop(lora_name, None)
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since registration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return

                    yield {
                        "status": "success",
                        "message": (
                            f"LoRA adapter '{lora_name}' "
                            f"{'hot-swapped' if is_hot_swap else 'loaded'} successfully"
                        ),
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                        "hot_swap": is_hot_swap,
                    }
                except Exception as e:
                    # Catch unexpected exceptions (e.g., from lora_name_to_id, engine calls)
                    # and clean up the capacity reservation to prevent ghost entries.
                    if capacity_reserved:
                        self._lora_state.loaded_loras.pop(lora_name, None)
                    logger.exception(f"Failed to load LoRA adapter: {e}")
                    yield {"status": "error", "message": str(e)}
                finally:
                    # Always release placeholder reservations even when the
                    # coroutine exits via cancellation/BaseException.
                    if capacity_reserved and not committed_lora_info:
                        existing = self._lora_state.loaded_loras.get(lora_name)
                        if existing is not None and existing.id == -1:
                            self._lora_state.loaded_loras.pop(lora_name, None)
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):
        """
        Unload a LoRA adapter dynamically from the vLLM's AsyncLLM engine.
        Expected request format:
        {
            "lora_name": str,
        }
        """
        try:
            try:
                lora_name = self._parse_lora_unload_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if the LoRA exists *after* waiting for any in-progress load.
                    lora = self._lora_state.loaded_loras.get(lora_name)
                    if lora is None:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self._lora_state.loaded_loras.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = lora.id

                    # Stop advertising the adapter before mutating engine or
                    # tracking state. Otherwise requests can still route here
                    # after _resolve_lora_request has forgotten the adapter and
                    # silently execute against the base model.
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Unregistering LoRA '{lora_name}' ModelDeploymentCard"
                        )
                        try:
                            await self._unregister_lora_discovery(lora_name)
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot unregister LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}"
                        )

                    # Prefill lifecycle registration is metadata-only, but
                    # vLLM may have activated the adapter lazily for an
                    # inference request. Remove only adapters known to have
                    # reached vLLM.
                    if lora_name in self._engine_loaded_loras:
                        try:
                            await self.engine_client.remove_lora(lora_id)
                        except Exception as e:
                            if not self._is_lora_not_loaded_error(e):
                                raise
                        self._engine_loaded_loras.discard(lora_name)
                    del self._lora_state.loaded_loras[lora_name]

                    logger.info(
                        f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )
                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Stripes are intentionally retained. Evicting a lock here
                    # can separate a waiting request from a later lifecycle op.
                    pass
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):
        """
        List all loaded LoRA adapters.
        Returns a dictionary of lora_name -> lora_id mappings.
        """
        try:
            loras = self._lora_state.list_lora_ids()
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    @staticmethod
    def _log_with_lora_context(
        message: str,
        request_id: str,
        lora_request=None,
        level: str = "debug",
        **kwargs,
    ) -> None:
        """
        Log a message with optional LoRA context.

        Args:
            message: Base message to log (can include {lora_info} placeholder)
            request_id: Request ID for correlation
            lora_request: Optional LoRA request object
            level: Log level ("debug" or "info")
            **kwargs: Additional format arguments for the message
        """
        if lora_request:
            lora_info = f" with LoRA {lora_request.lora_name}"
        else:
            lora_info = ""

        formatted_message = message.format(
            request_id=request_id,
            lora_info=lora_info,
            **kwargs,
        )

        if level == "info":
            logger.info(formatted_message)
        else:
            logger.debug(formatted_message)

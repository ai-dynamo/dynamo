# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import random
import socket
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import sglang as sgl
from sglang.srt.utils import get_local_ip_auto

from dynamo._core import Client, Component, Context, Endpoint
from dynamo.llm import (
    ModelInput,
    ModelType,
    lora_name_to_id,
    register_llm,
    unregister_llm,
)
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher

logger = logging.getLogger(__name__)

# LoRAManager singleton - initialized lazily when DYN_LORA_ENABLED is set
# None = not yet initialized, False = disabled/failed, LoRAManager = initialized
_lora_manager = None


def get_lora_manager():
    """Get the LoRAManager singleton, initializing it on first call if enabled."""
    global _lora_manager

    if _lora_manager is not None:
        return _lora_manager

    if os.environ.get("DYN_LORA_ENABLED", "").lower() in ("true", "1", "yes"):
        try:
            from dynamo.common.lora import LoRAManager

            _lora_manager = LoRAManager()
            logger.info("LoRAManager initialized successfully")
            return _lora_manager
        except Exception as e:
            logger.warning(
                f"Failed to initialize LoRAManager: {e}. URI-based LoRA loading will be disabled."
            )

    return None


class BaseWorkerHandler(ABC):
    """Abstract base class for SGLang worker handlers."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        prefill_client: Optional[Client] = None,
        generate_endpoint: Optional[Endpoint] = None,
    ) -> None:
        """Initialize base worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher for the worker.
            prefill_client: Optional client for prefill worker in disaggregated mode.
            generate_endpoint: Optional endpoint for LoRA registration.
        """
        self.component = component
        self.engine = engine
        self.config = config
        if publisher is not None:
            self.metrics_publisher = publisher.metrics_publisher
            self.kv_publisher = publisher.kv_publisher
        else:
            self.metrics_publisher = None
            self.kv_publisher = None
        self.prefill_client = prefill_client
        self.serving_mode = config.serving_mode
        self.skip_tokenizer_init = config.server_args.skip_tokenizer_init
        self.generate_endpoint = generate_endpoint

        # LoRA tracking
        self.lora_id_for_name: dict[str, int] = {}
        self.lora_name_to_path: dict[str, str] = {}

    @abstractmethod
    async def generate(self, request: Dict[str, Any], context: Context):
        """Generate response from request.

        Args:
            request: Request dict with input and parameters.
            context: Context object for cancellation handling.

        Yields:
            Response data (format varies by handler implementation).
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        pass

    async def load_lora(self, request: Optional[Dict[str, Any]] = None):
        """
        Load a LoRA adapter dynamically into the SGLang engine.

        Request format:
        {
            "lora_name": str,
            "source": {
                "uri": str  # e.g., "s3://bucket/path" or "file:///path"
            }
        }
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' and 'source.uri'",
                }
                return

            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Debug: Log the incoming request
            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")

            # Check for URI-based API format (source.uri)
            source = request.get("source")
            if not source or not isinstance(source, dict):
                yield {
                    "status": "error",
                    "message": "'source' object is required in request",
                }
                return

            lora_uri = source.get("uri")
            if not lora_uri:
                yield {
                    "status": "error",
                    "message": "'source.uri' is required in request",
                }
                return

            # Use LoRAManager to download from URI
            lora_manager = get_lora_manager()
            if lora_manager is None:
                yield {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }
                return

            logger.info(f"Downloading LoRA adapter: {lora_name} from {lora_uri}")
            download_result = await lora_manager.download_lora(lora_uri)

            if download_result["status"] != "success":
                yield {
                    "status": "error",
                    "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                }
                return

            lora_path = download_result["local_path"]
            logger.debug(f"LoRA downloaded to: {lora_path}")

            # Generate deterministic ID from lora_name before using it
            lora_id = lora_name_to_id(lora_name)

            # Add the LoRA to the SGLang engine via tokenizer_manager
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                # SGLang's load_lora_adapter expects lora_path as the positional argument
                await self.engine.tokenizer_manager.load_lora_adapter(lora_path)
            else:
                yield {
                    "status": "error",
                    "message": "SGLang engine does not support LoRA loading (tokenizer_manager not available)",
                }
                return

            # Track the LoRA
            self.lora_id_for_name[lora_name] = lora_id
            self.lora_name_to_path[lora_name] = lora_path
            logger.info(
                f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
            )

            # Publish LoRA as a ModelDeploymentCard
            if self.generate_endpoint is not None and self.config is not None:
                logger.debug(
                    f"Publishing LoRA '{lora_name}' ModelDeploymentCard to {self.generate_endpoint}"
                )
                try:
                    logger.debug(f"Publishing LoRA '{lora_name}' ModelDeploymentCard")

                    # Mark this as a LoRA in user_data
                    user_data = {
                        "lora_adapter": True,
                        "lora_id": lora_id,
                    }

                    await register_llm(
                        model_input=ModelInput.Tokens,
                        model_type=ModelType.Chat | ModelType.Completions,
                        endpoint=self.generate_endpoint,
                        model_path=self.config.server_args.model_path,
                        kv_cache_block_size=self.config.server_args.page_size,
                        user_data=user_data,
                        lora_name=lora_name,
                        base_model_path=self.config.server_args.model_path,
                    )
                    logger.info(
                        f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                    )
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                    )
                    logger.debug(f"Traceback: {traceback.format_exc()}")

                    # Rollback: remove the LoRA from the engine to maintain consistency
                    try:
                        logger.debug(
                            f"Rolling back: removing LoRA '{lora_name}' from engine"
                        )
                        # SGLang's unload_lora_adapter expects lora_path as the positional argument
                        await self.engine.tokenizer_manager.unload_lora_adapter(
                            lora_path
                        )
                        # Remove from tracking dictionaries
                        if lora_name in self.lora_id_for_name:
                            del self.lora_id_for_name[lora_name]
                        if lora_name in self.lora_name_to_path:
                            del self.lora_name_to_path[lora_name]
                        logger.debug(f"Successfully rolled back LoRA '{lora_name}'")
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                        )

                    # Return error status since registration failed
                    yield {
                        "status": "error",
                        "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                        "lora_name": lora_name,
                    }
                    return
            else:
                logger.debug(
                    f"Cannot publish LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}, config={self.config}"
                )

            yield {
                "status": "success",
                "message": f"LoRA adapter '{lora_name}' loaded successfully",
                "lora_name": lora_name,
                "lora_id": lora_id,
            }
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request: Optional[Dict[str, Any]] = None):
        """
        Unload a LoRA adapter dynamically from the SGLang engine.
        Expected request format:
        {
            "lora_name": str,
        }
        """
        try:
            if request is None:
                yield {
                    "status": "error",
                    "message": "Request is required with 'lora_name' field",
                }
                return
            lora_name = request.get("lora_name")
            if not lora_name:
                yield {
                    "status": "error",
                    "message": "'lora_name' is required in request",
                }
                return

            # Check if the LoRA exists
            if lora_name not in self.lora_id_for_name:
                yield {
                    "status": "error",
                    "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.lora_id_for_name.keys())}",
                }
                return

            logger.debug(f"Unloading LoRA adapter: {lora_name}")
            lora_id = self.lora_id_for_name[lora_name]
            lora_path = self.lora_name_to_path.get(lora_name)

            # Unload from SGLang engine
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                # SGLang's unload_lora_adapter expects lora_path as the positional argument
                await self.engine.tokenizer_manager.unload_lora_adapter(lora_path)
            else:
                yield {
                    "status": "error",
                    "message": "SGLang engine does not support LoRA unloading (tokenizer_manager not available)",
                }
                return

            # Remove from tracking dictionaries
            del self.lora_id_for_name[lora_name]
            if lora_name in self.lora_name_to_path:
                del self.lora_name_to_path[lora_name]

            # Unregister the LoRA model from the model registry
            if self.generate_endpoint is not None:
                logger.debug(f"Unregistering LoRA '{lora_name}' ModelDeploymentCard")
                try:
                    await unregister_llm(
                        endpoint=self.generate_endpoint,
                        lora_name=lora_name,
                    )
                    logger.info(
                        f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                    )
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                    )
                    logger.debug(f"Traceback: {traceback.format_exc()}")

                    # Rollback: re-add the LoRA to the engine to maintain consistency
                    try:
                        logger.debug(
                            f"Rolling back: re-adding LoRA '{lora_name}' to engine"
                        )
                        # SGLang's load_lora_adapter expects lora_path as the positional argument
                        await self.engine.tokenizer_manager.load_lora_adapter(lora_path)
                        # Re-add to tracking dictionaries
                        self.lora_id_for_name[lora_name] = lora_id
                        if lora_path:
                            self.lora_name_to_path[lora_name] = lora_path
                        logger.debug(f"Successfully rolled back LoRA '{lora_name}'")
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                        )

                    # Return error status since unregistration failed
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

            logger.info(
                f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
            )
            yield {
                "status": "success",
                "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                "lora_name": lora_name,
                "lora_id": lora_id,
            }
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request: Optional[Dict[str, Any]] = None):
        """
        List all loaded LoRA adapters.
        Returns a dictionary of lora_name -> lora_id mappings.
        """
        try:
            loras = dict(self.lora_id_for_name)
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    def _get_input_param(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate input parameter for SGLang engine.

        Args:
            request: Request dict with token_ids or messages.

        Returns:
            Dict with either input_ids or prompt for engine.
        """
        if self.skip_tokenizer_init:
            return {"input_ids": request["token_ids"]}
        else:
            # use sglang's chat templating itself but leave tokenization to the
            # interal engine's TokenizerManager
            prompt = self.engine.tokenizer_manager.tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True
            )
            return {"prompt": prompt}

    @staticmethod
    def _generate_bootstrap_room() -> int:
        """Generate a unique bootstrap room ID for disaggregated serving.

        Returns:
            Random 63-bit integer.
        """
        return random.randint(0, 2**63 - 1)

    @staticmethod
    def _get_bootstrap_info(engine: sgl.Engine) -> Tuple[str, int]:
        """Extract bootstrap host and port from SGLang engine.

        Args:
            engine: The SGLang engine instance.

        Returns:
            Tuple of (bootstrap_host, bootstrap_port).
        """
        inner_tm = engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_local_ip_auto()

        return bootstrap_host, bootstrap_port

    async def _handle_cancellation(
        self, request_id_future: asyncio.Future, context: Context
    ):
        """Background task to handle cancellation by monitoring context state.

        Args:
            request_id_future: Future that will be set with the SGLang request ID
                              when the first response arrives.
            context: Context object for cancellation handling.
        """
        try:
            logging.debug(f"Cancellation monitor started for Context: {context.id()}")

            # Always wait for the request ID to ensure we can abort the request
            sglang_request_id = await request_id_future
            logging.debug(
                f"Cancellation monitor received SGLang Request ID {sglang_request_id} for Context: {context.id()}"
            )
            logging.debug(f"Request ID future cancelled for Context: {context.id()}")

            await context.async_killed_or_stopped()

            logging.info(
                f"Cancellation signal received for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
            )

            # Call abort_request on the tokenizer_manager through the engine
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                logging.info(
                    f"Calling SGLang abort_request for Request ID {sglang_request_id}"
                )
                self.engine.tokenizer_manager.abort_request(
                    rid=sglang_request_id, abort_all=False
                )
                logging.info(f"Aborted Request ID: {context.id()}")
            else:
                logging.error(
                    f"SGLang tokenizer_manager not found for abort request: {context.id()}"
                )
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes
            request_id = "unknown"
            if request_id_future.done() and not request_id_future.cancelled():
                try:
                    request_id = request_id_future.result()
                except Exception:
                    pass
            logging.debug(
                f"Cancellation monitor task cancelled for SGLang Request ID {request_id}, Context: {context.id()}"
            )
            raise

    @asynccontextmanager
    async def _cancellation_monitor(
        self, request_id_future: asyncio.Future, context: Context
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Context manager for monitoring request cancellation.
        Automatically creates a background task to monitor for cancellation and
        cleans it up when the context exits.

        Args:
            request_id_future: Future that will be set with the SGLang request ID
                              when the first response arrives.
            context: Context object for cancellation handling

        Yields:
            asyncio.Task: The cancellation monitoring task being managed
        """
        logging.debug(f"Creating cancellation monitor task for Context: {context.id()}")

        # Start the cancellation monitoring task
        cancellation_task = asyncio.create_task(
            self._handle_cancellation(request_id_future, context)
        )

        try:
            yield cancellation_task
        finally:
            # Clean up the background cancellation task
            request_id = "unknown"
            if request_id_future.done() and not request_id_future.cancelled():
                try:
                    request_id = request_id_future.result()
                except Exception:
                    pass

            if not cancellation_task.done():
                logging.debug(
                    f"Cancelling cancellation monitor task for SGLang Request ID {request_id}, Context: {context.id()}"
                )
                cancellation_task.cancel()
                try:
                    await cancellation_task
                except asyncio.CancelledError:
                    pass
            else:
                logging.debug(
                    f"Cancellation monitor task already completed for SGLang Request ID {request_id}, Context: {context.id()}"
                )

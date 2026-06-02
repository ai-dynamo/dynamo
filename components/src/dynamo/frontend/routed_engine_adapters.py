# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology-specific wrappers for frontend routed-engine calls."""

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from dynamo.common.global_router_protocol import (
    GLOBAL_ROUTER_ACTION_EXHAUSTED,
    GLOBAL_ROUTER_ACTION_RETRY,
    GLOBAL_ROUTER_RETRY_ATTEMPT_KEY,
    get_global_router_control,
)

if TYPE_CHECKING:
    from dynamo.llm import RoutedEngine
else:
    RoutedEngine = Any

logger = logging.getLogger(__name__)

ROUTED_ENGINE_ADAPTER_DEFAULT = "default"
ROUTED_ENGINE_ADAPTER_GLOBAL_ROUTER = "global-router"
VALID_ROUTED_ENGINE_ADAPTERS = {
    ROUTED_ENGINE_ADAPTER_DEFAULT,
    ROUTED_ENGINE_ADAPTER_GLOBAL_ROUTER,
}

_MAX_GLOBAL_ROUTER_RETRY_CONTROLS = 32


class _GlobalRouterControlError(RuntimeError):
    pass


class _GlobalRouterRetryExhausted(_GlobalRouterControlError):
    pass


def wrap_routed_engine(config: Any, routed_engine: RoutedEngine) -> RoutedEngine:
    adapter = getattr(config, "routed_engine_adapter", ROUTED_ENGINE_ADAPTER_DEFAULT)
    if adapter == ROUTED_ENGINE_ADAPTER_GLOBAL_ROUTER:
        return GlobalRouterRoutedEngineAdapter(routed_engine)
    return routed_engine


class GlobalRouterRoutedEngineAdapter:
    """Frontend-side retry loop for direct responses through a global router.

    The global router handles one pool attempt per request when this adapter is
    enabled. If that dispatch fails before a delegated response is established,
    the global router emits a control item instructing the frontend to reissue
    the same request with the next retry attempt.
    """

    def __init__(self, routed_engine: RoutedEngine):
        self._routed_engine = routed_engine

    async def generate(self, request: Any, **kwargs: Any) -> AsyncIterator[Any]:
        return self._generate_with_global_router_retries(request, **kwargs)

    async def _generate_with_global_router_retries(
        self, request: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        attempt = 0
        retry_controls = 0

        while True:
            yielded_output = False
            next_attempt = None
            retry_error = None

            try:
                stream = await self._routed_engine.generate(
                    _request_for_retry_attempt(request, attempt), **kwargs
                )

                async for output in stream:
                    data = output.data() if hasattr(output, "data") else output
                    control = get_global_router_control(data)
                    if control is not None:
                        if yielded_output:
                            raise RuntimeError(
                                "global router retry control received after response "
                                "streaming started"
                            )
                        action = control.get("action")
                        if action == GLOBAL_ROUTER_ACTION_EXHAUSTED:
                            raise _GlobalRouterRetryExhausted(
                                _retry_exhausted_message(control)
                            )
                        next_attempt = _next_retry_attempt(control)
                        logger.warning(
                            "Retrying global-router request via next pool: "
                            "attempt=%s next_attempt=%s failed_namespace=%s "
                            "next_namespace=%s error=%s",
                            attempt,
                            next_attempt,
                            control.get("failed_namespace"),
                            control.get("next_namespace"),
                            control.get("error"),
                        )
                        break

                    yielded_output = True
                    yield output
            except _GlobalRouterControlError:
                raise
            except Exception as exc:
                if yielded_output:
                    raise
                next_attempt = attempt + 1
                retry_error = exc
                logger.warning(
                    "Retrying global-router request after pre-output delegated "
                    "response failure: attempt=%s next_attempt=%s error=%s",
                    attempt,
                    next_attempt,
                    exc,
                )

            if next_attempt is None:
                return
            if next_attempt <= attempt:
                raise RuntimeError(
                    "global router retry attempt did not increase: "
                    f"attempt={attempt} next_attempt={next_attempt}"
                )

            retry_controls += 1
            if retry_controls > _MAX_GLOBAL_ROUTER_RETRY_CONTROLS:
                raise RuntimeError(
                    "global router retry control loop exceeded limit"
                ) from retry_error
            attempt = next_attempt


def _retry_exhausted_message(control: Any) -> str:
    error = control.get("error")
    if error:
        return str(error)
    return (
        "global router retry attempts exhausted: "
        f"request_type={control.get('request_type')} "
        f"retry_attempt={control.get('retry_attempt')}"
    )


def _request_for_retry_attempt(request: Any, retry_attempt: int) -> Any:
    if not isinstance(request, dict):
        raise TypeError("global-router routed-engine adapter requires dict requests")
    next_request = dict(request)
    routing = dict(request.get("routing") or {})
    routing[GLOBAL_ROUTER_RETRY_ATTEMPT_KEY] = retry_attempt
    next_request["routing"] = routing
    return next_request


def _next_retry_attempt(control: Any) -> int:
    if control.get("action") != GLOBAL_ROUTER_ACTION_RETRY:
        raise _GlobalRouterControlError(
            f"unsupported global router control action: {control!r}"
        )
    value = control.get("next_retry_attempt")
    if isinstance(value, bool):
        raise _GlobalRouterControlError(
            f"invalid global router next retry attempt: {value!r}"
        )
    try:
        next_attempt = int(value)
    except (TypeError, ValueError) as exc:
        raise _GlobalRouterControlError(
            f"invalid global router next retry attempt: {value!r}"
        ) from exc
    if next_attempt < 0:
        raise _GlobalRouterControlError(
            f"invalid global router next retry attempt: {value!r}"
        )
    return next_attempt

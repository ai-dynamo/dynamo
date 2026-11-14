# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, versioned RL control plane composed into the vLLM worker."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.llm.exceptions import InvalidArgument

from .generation_policy import StrictRLGenerationObserver
from .weight_transfer import (
    WeightTransferSession,
    WeightTransferStatus,
    WeightTransferValidationError,
)

# The backend is optional in collection environments; strict startup is not.
try:
    from vllm.distributed.weight_transfer.base import (
        WeightTransferInitRequest,
        WeightTransferUpdateRequest,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "vllm.distributed.weight_transfer",
        "vllm.distributed.weight_transfer.base",
    }:
        raise
    WeightTransferInitRequest = None
    WeightTransferUpdateRequest = None

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.inputs import PromptType
    from vllm.sampling_params import SamplingParams


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class StrictRLStatus(WeightTransferStatus):
    model_max_len: int
    device_uuids: tuple[str, ...]
    status: str = "ok"


class StrictRLWeightTransferController:
    """Own admission, sequenced installs and the only strict mutation routes."""

    def __init__(
        self,
        engine_client: "EngineClient",
        model_max_len: int,
        pause_lock: asyncio.Lock,
        shutdown_on_engine_dead: Callable[[EngineDeadError], None],
        *,
        device_uuids: tuple[str, ...],
    ) -> None:
        if WeightTransferInitRequest is None:
            raise ValueError("strict RL requires vLLM weight-transfer support")
        if type(model_max_len) is not int or model_max_len <= 0:
            raise ValueError("strict RL requires a positive model_max_len")
        self.engine_client = engine_client
        self.session = WeightTransferSession()
        self._model_max_len = model_max_len
        self._lock = pause_lock
        self._shutdown_on_engine_dead = shutdown_on_engine_dead
        self._device_uuids = device_uuids

    def routes(self) -> dict[str, Callable[[dict], Awaitable[dict]]]:
        """Advertise only the strict protocol, without legacy mutation aliases."""
        return {
            "pause_generation": self.pause_generation,
            "resume_generation": self.resume_generation,
            "flush_cache": self.flush_cache,
            "get_weight_version": self.get_weight_version,
            "get_weight_checksums": self.get_weight_checksums,
            "update/init_weight_transfer_engine": self.init_weight_transfer_engine,
            "update/begin_weight_update": self.begin_weight_update,
            "update/start_weight_update": self.start_weight_update,
            "update/update_weights": self.update_weights,
            "update/finish_weight_update": self.finish_weight_update,
        }

    def request_observer(
        self,
        sampling_params: "SamplingParams",
        prompt: "PromptType | None",
        request_id: str,
    ) -> StrictRLGenerationObserver:
        """Bind validation to the same native engine that owns the policy."""
        if not isinstance(prompt, dict) or "prompt_token_ids" not in prompt:
            raise InvalidArgument("strict RL requires a token prompt")
        return StrictRLGenerationObserver(
            self.engine_client,
            sampling_params,
            prompt["prompt_token_ids"],
            request_id,
            self.session,
        )

    def versioned_route(
        self, callback: Callable[[dict], Awaitable[dict]]
    ) -> Callable[[dict], Awaitable[dict]]:
        """Attach the strict contract to canonical liveness and abort responses."""

        async def route(body: dict) -> dict:
            response = await callback(body)
            if response["status"] != "ok":
                return response
            return {**response, **self._status()}

        return route

    def _status(self) -> dict:
        snapshot = self.session.snapshot()
        result = StrictRLStatus(
            **asdict(snapshot),
            model_max_len=self._model_max_len,
            device_uuids=self._device_uuids,
        )
        payload = asdict(result)
        payload["device_uuids"] = list(result.device_uuids)
        return payload

    async def _locked(self, operation: Callable[[], Awaitable[None]]) -> dict:
        """Return protocol rejections but fence and propagate native failures."""
        async with self._lock:
            try:
                await operation()
                return self._status()
            except EngineDeadError as exc:
                self.session.fail()
                self._shutdown_on_engine_dead(exc)
                raise
            except WeightTransferValidationError as exc:
                return {"status": "error", "message": str(exc)}
            except Exception as exc:
                self.session.fail()
                logger.warning("Strict RL control failed (%s)", type(exc).__name__)
                raise RuntimeError(
                    "strict RL engine operation failed; restart worker"
                ) from exc
            except BaseException:
                self.session.fail()
                raise

    @staticmethod
    def _empty(body: Any) -> None:
        if not isinstance(body, dict) or body:
            raise WeightTransferValidationError(
                "request body must be an empty JSON object"
            )

    async def pause_generation(self, body: dict) -> dict:
        if (
            not isinstance(body, dict)
            or set(body) - {"mode", "clear_cache"}
            or body.get("mode") not in ("wait", "abort")
            or body.get("clear_cache", False) is not False
        ):
            return {
                "status": "error",
                "message": "strict RL pause requires wait/abort and clear_cache=false",
            }

        async def pause() -> None:
            await self.session.pause(self.engine_client.pause_generation, body["mode"])

        return await self._locked(pause)

    async def resume_generation(self, body: dict) -> dict:
        async def resume() -> None:
            self._empty(body)
            await self.session.resume(self.engine_client.resume_generation)

        return await self._locked(resume)

    async def flush_cache(self, body: dict) -> dict:
        async def flush() -> None:
            self._empty(body)
            await self.session.flush(
                lambda: self.engine_client.reset_prefix_cache(reset_connector=True)
            )

        return await self._locked(flush)

    async def get_weight_version(self, body: dict) -> dict:
        # Validate before any collective; diagnostics have a separate route.
        if not isinstance(body, dict) or body:
            return {
                "status": "error",
                "message": "request body must be an empty JSON object",
            }

        async def readback() -> None:
            snapshot = self.session.snapshot()
            if snapshot.install_complete and (
                await self.engine_client.get_weight_version() != snapshot.version
            ):
                self.session.fail()
                raise RuntimeError(
                    "engine policy version does not match installed session"
                )

        return await self._locked(readback)

    async def get_weight_checksums(self, body: dict) -> dict:
        checksums: dict[str, str] = {}

        async def collect() -> None:
            nonlocal checksums
            self._empty(body)
            values = await self.engine_client.compute_weight_checksums()
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], dict)
                or not values[0]
                or any(
                    not isinstance(name, str) or not isinstance(digest, str)
                    for name, digest in values[0].items()
                )
            ):
                raise RuntimeError("invalid weight-checksum metadata")
            checksums = values[0]

        response = await self._locked(collect)
        return (
            {**response, "weight_checksums": checksums}
            if response["status"] == "ok"
            else response
        )

    async def _initialize(self, init_info: dict) -> None:
        await self.engine_client.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info=init_info)
        )

    async def _apply(self, update_info: dict) -> None:
        await self.engine_client.update_weights(
            WeightTransferUpdateRequest(update_info=update_info)
        )

    async def _synchronize(self) -> None:
        await self.engine_client.collective_rpc("synchronize_device")

    async def init_weight_transfer_engine(self, body: dict) -> dict:
        async def initialize() -> None:
            await self.session.initialize(body, self._initialize)

        return await self._locked(initialize)

    async def begin_weight_update(self, body: dict) -> dict:
        async def begin() -> None:
            self.session.begin(body)

        return await self._locked(begin)

    async def start_weight_update(self, body: dict) -> dict:
        async def start() -> None:
            await self.session.start(body, self.engine_client.start_weight_update)

        return await self._locked(start)

    async def update_weights(self, body: dict) -> dict:
        async def update() -> None:
            await self.session.update(body, self._apply, self._synchronize)

        return await self._locked(update)

    async def finish_weight_update(self, body: dict) -> dict:
        async def finish() -> None:
            await self.session.finish(
                body,
                self.engine_client.finish_weight_update,
                self._synchronize,
            )

        return await self._locked(finish)

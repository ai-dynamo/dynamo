# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory-selected admission and request-local generation validation."""

from typing import TYPE_CHECKING, Protocol

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from dynamo.common.backend.logprobs import validate_sampled_logprobs
from dynamo.llm.exceptions import WorkerUnavailable

from .strict_termination import StrictRLTermination
from .weight_transfer import WeightTransferSession

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.inputs import PromptType


class RequestObserver(Protocol):
    def observe(self, response: RequestOutput) -> None:
        ...

    def finish(self) -> None:
        ...


class GenerationPolicy(Protocol):
    def request_observer(
        self,
        sampling_params: SamplingParams,
        prompt: "PromptType | None",
        request_id: str,
    ) -> RequestObserver:
        ...


class _PassthroughObserver:
    __slots__ = ()

    def observe(self, response: RequestOutput) -> None:
        return None

    def finish(self) -> None:
        return None


_PASSTHROUGH_OBSERVER = _PassthroughObserver()


class _PassthroughPolicy:
    __slots__ = ()

    def request_observer(
        self,
        sampling_params: SamplingParams,
        prompt: "PromptType | None",
        request_id: str,
    ) -> RequestObserver:
        return _PASSTHROUGH_OBSERVER


PASSTHROUGH_GENERATION_POLICY: GenerationPolicy = _PassthroughPolicy()


class StrictRLGenerationObserver:
    """Validate every native DELTA before it can become a rollout response."""

    def __init__(
        self,
        engine: "EngineClient",
        sampling_params: SamplingParams,
        prompt_token_ids: list[int],
        request_id: str,
        session: WeightTransferSession,
    ) -> None:
        self._session = session
        try:
            self._version = session.require_generation()
        except ValueError as exc:
            raise WorkerUnavailable(str(exc)) from exc
        self._template = StrictRLTermination.from_engine(
            engine, sampling_params, prompt_token_ids, request_id
        )
        self._require_logprobs = sampling_params.logprobs is not None
        self._choice_count = sampling_params.n
        self._states: dict[int, StrictRLTermination] = {}
        self._observed_prompt: list[int] | None = None

    def observe(self, response: RequestOutput) -> None:
        self._require_version()
        if (
            response.prompt_token_ids is not self._observed_prompt
            or self._observed_prompt is None
        ):
            if (
                tuple(response.prompt_token_ids or ())
                != self._template.prompt_token_ids
            ):
                raise RuntimeError(
                    "strict RL engine prompt differs from the admitted prompt"
                )
            self._observed_prompt = response.prompt_token_ids
        for output in response.outputs:
            index = output.index
            if type(index) is not int or not 0 <= index < self._choice_count:
                raise RuntimeError("strict RL received an invalid choice index")
            try:
                state = self._states.get(index, self._template).observe(output)
                if self._require_logprobs:
                    validate_sampled_logprobs(output)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            self._states = {**self._states, index: state}

    def finish(self) -> None:
        if len(self._states) != self._choice_count or not all(
            state.finished for state in self._states.values()
        ):
            raise RuntimeError("strict RL stream ended without terminal evidence")

    def _require_version(self) -> None:
        try:
            self._session.require_version(self._version)
        except ValueError as exc:
            if self._states:
                raise RuntimeError(str(exc)) from exc
            raise WorkerUnavailable(str(exc)) from exc

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo-to-vLLM decoder implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from enum import Enum
from typing import Any, cast

from vllm.config import ModelConfig, VllmConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.common.backend import GenerateChunk, GenerateRequest
from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.llm.exceptions import InvalidArgument

from .decoder_output import ExtractLogprobs, _VllmDecodeOutputAdapter
from .decoder_sampling import build_sampling_params

GenerateFactory = Callable[[Any], AsyncIterator[RequestOutput]]
GenerationAdmission = Callable[[Any, GenerateFactory], AsyncIterator[RequestOutput]]
_LORA_UNSET = object()


class DecodeOutputMode(Enum):
    """Native output shape requested by a Dynamo decoder adapter."""

    STREAM_DELTA = RequestOutputKind.DELTA
    FINAL_ONLY = RequestOutputKind.FINAL_ONLY


class VllmDecoderRuntime:
    """Own and drive one native vLLM decoder for Dynamo adapters."""

    def __init__(
        self,
        *,
        engine: AsyncLLM,
        vllm_config: VllmConfig,
        default_sampling_params: Mapping[str, Any] | None,
    ) -> None:
        self._engine: AsyncLLM | None = engine
        self._vllm_config = vllm_config
        self._default_sampling_params = dict(default_sampling_params or {})

    @property
    def engine(self) -> AsyncLLM:
        """Return the native engine while this runtime is active."""

        if self._engine is None:
            raise RuntimeError("VllmDecoderRuntime is shut down")
        return self._engine

    @property
    def vllm_config(self) -> VllmConfig:
        return self._vllm_config

    @property
    def model_config(self) -> ModelConfig:
        return self._vllm_config.model_config

    @property
    def default_sampling_params(self) -> Mapping[str, Any]:
        return self._default_sampling_params

    def prepare_sampling_params(
        self,
        request: GenerateRequest | Mapping[str, Any],
        *,
        enable_rl: bool = False,
        prompt: Any | None = None,
        output_mode: DecodeOutputMode = DecodeOutputMode.STREAM_DELTA,
    ) -> SamplingParams:
        """Translate a Dynamo request and enforce the resolved prompt budget."""

        request_dict = cast(dict[str, Any], request)
        prepared_token_ids = (
            prompt.get("prompt_token_ids") if isinstance(prompt, Mapping) else None
        )
        prompt_token_count = len(
            prepared_token_ids
            if prepared_token_ids is not None
            else request_dict.get("token_ids") or []
        )

        available_tokens = self.model_config.max_model_len - prompt_token_count
        if available_tokens < 1:
            raise InvalidArgument(
                "prepared decoder prompt leaves no room for generation: "
                f"prompt_tokens={prompt_token_count}, "
                f"max_model_len={self.model_config.max_model_len}"
            )
        requested_max_tokens = (request_dict.get("stop_conditions") or {}).get(
            "max_tokens"
        )
        if requested_max_tokens is not None and requested_max_tokens > available_tokens:
            raise InvalidArgument(
                "requested max_tokens exceeds the prepared prompt budget: "
                f"max_tokens={requested_max_tokens}, available={available_tokens}"
            )

        sampling_params = build_sampling_params(
            request_dict,
            dict(self._default_sampling_params),
            model_max_len=self.model_config.max_model_len,
            enable_rl=enable_rl,
            prompt_token_count_override=prompt_token_count,
        )
        sampling_params.detokenize = False
        sampling_params.output_kind = output_mode.value
        return sampling_params

    async def decode(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        engine_options: Mapping[str, Any] | None = None,
        lora_request: Any = _LORA_UNSET,
        admission: GenerationAdmission | None = None,
        extract_logprobs: ExtractLogprobs | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Run native generation and emit canonical Dynamo token chunks."""

        options = dict(engine_options or {})

        def create_generator(
            admitted_lora_request: Any,
        ) -> AsyncIterator[RequestOutput]:
            admitted_options = dict(options)
            if admitted_lora_request is not _LORA_UNSET:
                admitted_options["lora_request"] = admitted_lora_request
            return self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                **admitted_options,
            )

        if admission is None:
            native_stream = create_generator(lora_request)
        else:
            native_stream = admission(lora_request, create_generator)

        output_adapter = _VllmDecodeOutputAdapter(
            sampling_params,
            tokenizer=getattr(self.engine, "tokenizer", None),
            extract_logprobs=extract_logprobs or self._extract_logprobs,
        )
        async for response in native_stream:
            for chunk in output_adapter.convert(response):
                yield cast(GenerateChunk, chunk)
            # Preserve the production handler's terminal handling for a native
            # response with no choices. The adapter emits one error chunk; no
            # later engine responses should leak through for the same request.
            if not response.outputs:
                break

    @staticmethod
    def _extract_logprobs(
        output: Any,
        num_output_tokens_so_far: int,
        tokenizer: Any = None,
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        return _shared_logprobs.extract_from_completion_output(
            output,
            num_output_tokens_so_far,
            tokenizer=tokenizer,
            fallback_to_first_on_missing=True,
            include_bytes=True,
        )

    async def abort(self, request_id: str) -> None:
        """Abort an active request if the runtime has not been shut down."""

        engine = self._engine
        if engine is not None:
            await engine.abort(request_id)

    def shutdown(self) -> None:
        """Shut down the native engine at most once."""

        engine = self._engine
        self._engine = None
        if engine is not None:
            engine.shutdown(timeout=self._vllm_config.shutdown_timeout)
